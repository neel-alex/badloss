"""Neural Cleanse (Wang et al., 2019), adapted to filter the training set and retrain.

Reverse-engineer a minimal mask + trigger per class; classes with anomalously small masks (MAD anomaly index)
are treated as attacked. Training examples that strongly activate the trigger's most affected neurons are
removed, with a threshold set by the false-positive rate on the defender's clean probes.
"""
import numpy as np
import torch
from tqdm import tqdm


def apply_mask_and_trigger(batch, mask, trigger):
    return batch * (1 - mask) + mask * trigger


def train_cleanse(exp, loader, mask, trigger, optimizer, target_class, l1_penalty, log_interval=5):
    """One epoch of mask/trigger optimization towards target_class; returns the adjusted L1 penalty."""
    optimizer.zero_grad()
    pbar = tqdm(loader)
    total_in_cls = 0
    for batch_idx, ((data, target), ex_idx) in enumerate(pbar):
        data = data.to(exp.device)
        triggered_data = apply_mask_and_trigger(data, mask, trigger)
        cleanse_target = torch.full(target.shape, target_class, device=exp.device)
        optimizer.zero_grad()

        output = exp.model(triggered_data)
        predictions = torch.argmax(output, 1)
        total_in_cls += (predictions == target_class).sum().item()

        trigger_loss = exp.criterion(output, cleanse_target).mean()
        l1_loss = torch.norm(mask, p=1) * l1_penalty
        loss = trigger_loss + l1_loss

        loss.backward()
        optimizer.step()

        mask.data.clamp_(0, 1)
        trigger.data.clamp_(0, 1)

        if batch_idx % log_interval == 0:
            pbar.set_description(f"Loss: {float(loss.detach()):.4f}")
    print(f"Classifies {total_in_cls} as {target_class}")
    # Simplified penalty schedule (the paper's is fiddly): keep ~99% of the training set flipped to the target
    if total_in_cls < 0.99 * len(loader.sampler):
        l1_penalty /= 2  # Not enough flipped: allow a larger mask
    if total_in_cls > 0.99 * len(loader.sampler):
        l1_penalty *= 2  # More than enough flipped: keep the mask small
    return l1_penalty


def get_indices_for_thresh(fpr_thresh, clean_probe_indices, attacked_classes, poison_acts_by_class,
                           indices_to_check, clean_indices):
    """Flag examples whose poisoned-neuron activation exceeds a threshold that rejects at most fpr_thresh of the
    clean probes, for any attacked class."""
    upper_limit = 1 + int(len(clean_probe_indices) * fpr_thresh)
    rejected_indices = []
    for i, atk_class in enumerate(attacked_classes):
        poison_acts = poison_acts_by_class[i]
        reject_thresh = poison_acts[indices_to_check].sort()[0][-upper_limit]
        rejected_indices.append((poison_acts > reject_thresh).nonzero()[:, 0])
    if not rejected_indices:  # No anomalous classes detected
        return np.array([], dtype=int)
    rejected_indices = torch.hstack(rejected_indices).unique()
    return clean_indices[rejected_indices.cpu()].numpy()


def run(exp):
    args = exp.args
    exp.pretrain()
    exp.report_attacked_model()
    exp.wandb_prefix = "retraining_"
    model = exp.model

    def detect():
        # Optionally subsample classes (always keeping the attack targets) and the data the masks are optimized on
        rng = np.random.default_rng(exp.seed)
        classes = list(range(exp.num_classes))
        if args.nc_class_frac < 1:
            targets = sorted({int(t) for t in exp.attack_targets.values()})
            others = [c for c in classes if c not in targets]
            num_others = max(0, round(args.nc_class_frac * exp.num_classes) - len(targets))
            classes = sorted(targets + [int(c) for c in rng.choice(others, size=num_others, replace=False)])
            print(f"Neural Cleanse on {len(classes)} classes (attack targets: {targets})")
        cleanse_loader = exp.train_loader_wo_aug
        if args.nc_data_frac < 1:
            subset = rng.choice(exp.comb_train_indices, size=round(args.nc_data_frac * len(exp.comb_train_indices)),
                                replace=False)
            cleanse_loader = exp.loader(subset, dataset=exp.new_idx_loader_wo_aug.dataset)

        masks, norms, triggers = {}, [], {}
        mask_shape = exp.img_size[-1:] + exp.img_size[:-1]  # CHW
        for cls in classes:
            l1_penalty = 1.0
            mask = torch.nn.Parameter(torch.rand(size=mask_shape, device=exp.device))
            trigger = torch.nn.Parameter(torch.rand(size=mask_shape, device=exp.device))
            cleanse_opt = torch.optim.Adam((mask, trigger))
            for _ in range(args.nc_cleanse_epochs):
                l1_penalty = train_cleanse(exp, cleanse_loader, mask, trigger, cleanse_opt, cls, l1_penalty)

            masks[cls] = mask.detach()
            triggers[cls] = trigger.detach()
            norms.append(torch.norm(mask, p=1))
            print(f"Trained mask for class {cls}. Final l1 penalty: {l1_penalty}, "
                  f"final mask magnitude: {mask.norm(p=1)}")

        # Median absolute deviation (1.4826 makes it consistent with the std for normal data)
        norms = torch.tensor(norms)
        median = norms.median()
        absolute_deviations = (norms - median).abs()
        mad = 1.4826 * absolute_deviations.median()
        anomaly_index = absolute_deviations / mad

        flagged = (anomaly_index >= args.nc_anomaly_threshold).nonzero()[:, 0]
        print("Skipping detected attacks where norm is above median.")
        flagged = flagged[(norms[flagged] <= median).nonzero()[:, 0]].tolist()  # Positions in `classes`
        attacked_classes = [classes[i] for i in flagged]

        clean_activations, clean_indices, _, _ = exp.last_layer_activations(model, exp.new_idx_loader_wo_aug)
        clean_order = clean_indices.argsort()
        clean_activations, clean_indices = clean_activations[clean_order], clean_indices[clean_order]
        # Positions of the clean probes (they are ordinary training examples for this defense)
        clean_probe_indices = np.array(exp.probe_imgs.idx)
        indices_to_check = torch.isin(clean_indices, torch.tensor(clean_probe_indices)).nonzero()[:, 0]

        poison_acts_by_class = []
        for pos, atk_class in zip(flagged, attacked_classes):
            mask, trigger = masks[atk_class], triggers[atk_class]
            dirty_activations, dirty_indices, _, dirty_predictions = exp.last_layer_activations(
                model, exp.new_idx_loader_wo_aug, masking_op=lambda img: apply_mask_and_trigger(img, mask, trigger))
            attack_success = (dirty_predictions == atk_class).sum()
            print(f"Anomalous class {atk_class} with index {anomaly_index[pos].item():.3f} "
                  f"has mask magnitude {norms[pos].item():.3f} which classifies {attack_success.item()} as {atk_class}")
            # Align triggered activations with the clean ones by example
            dirty_order = dirty_indices.argsort()
            dirty_activations, dirty_indices = dirty_activations[dirty_order], dirty_indices[dirty_order]

            # Neurons most increased by the trigger; their mean activation on the clean image is the poison score
            mean_activation_diff = (dirty_activations - clean_activations).mean(axis=0)
            neurons_to_select = int(len(mean_activation_diff) * args.nc_neuron_frac)
            poisoned_neurons = mean_activation_diff.sort()[1][-neurons_to_select:]
            poison_acts_by_class.append(clean_activations[:, poisoned_neurons].mean(axis=1))

        fpr_thresh = args.nc_fpr / max(len(attacked_classes), 1)
        auc_idx = [get_indices_for_thresh(t, clean_probe_indices, attacked_classes, poison_acts_by_class,
                                          indices_to_check, clean_indices)
                   for t in np.geomspace(fpr_thresh / 20, 0.8, num=50)]
        exp.report_detection_auc("NC", auc_idx)

        identified_indices = get_indices_for_thresh(fpr_thresh, clean_probe_indices, attacked_classes,
                                                    poison_acts_by_class, indices_to_check, clean_indices)
        exp.report_detection(identified_indices)
        return identified_indices

    identified_indices = exp.detect(detect)
    if identified_indices is not None:
        exp.retrain(identified_indices)
    print("Done with nc")
