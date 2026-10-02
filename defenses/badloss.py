"""BaDLoss: Backdoor Detection via Loss Dynamics (arXiv:2408.13221).

Track each training example's loss (or correct-class probability) over a short training run, score it by its
distance to the nearest trajectories of the defender's bona fide clean probes, drop the most anomalous fraction
and retrain from scratch.
"""
import os
import pickle
from collections import Counter

import numpy as np
import sklearn.neighbors
import torch
from sklearn.metrics import roc_curve, auc

from torch_utils import train, test, collect_losses


def collect_trajectories(exp):
    """Train the attacked model for --badloss_pretrain_epochs, recording per-example loss and correct-class
    probability on the (un-augmented) combined training set after every epoch."""
    args = exp.args
    model_file = os.path.join(exp.model_dir, f"model_{args.dataset}.pth")
    data_file = os.path.join(exp.model_collection_dir, f"stats_{args.dataset}.pkl")

    if os.path.exists(model_file):
        assert os.path.exists(data_file)
        print("Data files already found. Loading data from saved checkpoints.")
        exp.model.load_state_dict(torch.load(model_file, map_location=exp.device))
        with open(data_file, "rb") as f:
            return pickle.load(f)

    num_examples = len(exp.new_idx_loader.dataset)
    losses = torch.zeros((num_examples, args.badloss_pretrain_epochs))
    correct_class_probs = torch.zeros((num_examples, args.badloss_pretrain_epochs))

    loader = exp.new_idx_loader if args.badloss_pretrain_augment else exp.new_idx_loader_wo_aug
    for epoch in range(args.badloss_pretrain_epochs):
        train(exp.model, exp.device, loader, exp.optimizer, exp.criterion)
        losses[:, epoch], correct_class_probs[:, epoch] = collect_losses(exp.model, exp.device,
                                                                         exp.new_idx_loader_wo_aug, exp.criterion)
        if epoch % 5 == 4:
            exp.evaluate_asr(exp.model, exp.criterion)
        exp.lr_scheduler.step()

    exp.evaluate_asr(exp.model, exp.criterion)
    stats = {'losses': losses, 'probs': correct_class_probs, 'probe_id': exp.dataset_probe_identity}
    if exp.main_proc:
        torch.save(exp.model.state_dict(), model_file)
        with open(data_file, "wb") as f:
            pickle.dump(stats, f, protocol=pickle.HIGHEST_PROTOCOL)
    return stats


def filter_epochs(trajectories):
    """Indices of epochs to keep: drop any epoch whose average loss exceeds twice the average of the previous
    three kept epochs (loss spikes are common in the multi-attack setting)."""
    epoch_avg_loss = trajectories.mean(axis=0)
    final_avgs = epoch_avg_loss[:3].tolist()
    epochs_to_keep = [0, 1, 2]
    for i in range(3, len(epoch_avg_loss)):
        if epoch_avg_loss[i] < 2 * sum(final_avgs[-3:]) / 3:
            epochs_to_keep.append(i)
            final_avgs.append(epoch_avg_loss[i])
    print(epochs_to_keep, final_avgs)
    return epochs_to_keep


def anomaly_scores(trajectories, clean_trajectories, k, eps, log_scores=True):
    """Score each trajectory by its mean distance to the k nearest clean trajectories, normalized to [0, 1]."""
    clf = sklearn.neighbors.KNeighborsClassifier(len(clean_trajectories))
    clf.fit(clean_trajectories, np.zeros(len(clean_trajectories), dtype=int))
    dists = clf.kneighbors(trajectories, n_neighbors=k)[0]
    mean_dists = dists.mean(axis=1)
    if log_scores:
        log_dists = np.log(eps + mean_dists)
        zero_min_dists = log_dists - log_dists.min()
        return zero_min_dists / zero_min_dists.max()
    return (mean_dists - mean_dists.min()) / (mean_dists - mean_dists.min()).max()


def run(exp):
    args = exp.args
    stats = collect_trajectories(exp)
    exp.report_attacked_model()

    losses = stats['losses'] if args.badloss_metric == 'loss' else stats['probs']
    dataset_probe_identity = np.array(exp.dataset_probe_identity)
    clean_idx = np.where(dataset_probe_identity == 'clean')[0]
    print("Training the trajectory classifier...")

    # Rows that were never visited: original positions of examples replaced by their probe/poison copies
    missing_vals = (losses == 0).all(dim=1)
    missing_vals_idx = torch.nonzero(missing_vals).squeeze()
    available_ex = ~missing_vals
    print(f"!! Total loss traj len: {len(losses)} / Modified examples identified: {sum(missing_vals)}")

    if not args.badloss_no_epoch_filter:
        losses = losses[:, filter_epochs(losses[available_ex])]
    probs = anomaly_scores(losses[available_ex], losses[clean_idx], args.badloss_k, args.badloss_eps,
                           log_scores=not args.badloss_linear_scores)

    # Convert scores to ranks in [0, 1], so that thresholding removes a fixed fraction of examples
    ranks = np.zeros(len(probs))
    ranks[probs.argsort()] = np.linspace(0, 1, len(probs))

    x, y, _ = roc_curve(np.array([1 if 'val' in x else 0 for x in dataset_probe_identity])[available_ex], ranks)
    print("AUC:", auc(x, y))
    exp.record_detection(auc=auc(x, y))
    poison_scores = np.zeros(len(losses), dtype=ranks.dtype)
    poison_scores[available_ex] = ranks
    poison_scores[missing_vals] = 1.1  # Always removed

    print("!! Including training probe examples with their clean labels for retraining...")
    probe_new_idx = [i for i, x in enumerate(dataset_probe_identity) if x in {"backdoor", "clean"}]
    assert len(probe_new_idx) == len(exp.train_probes_idx), f"{len(probe_new_idx)} == {len(exp.train_probes_idx)}"
    # Retrain on the probes' original training-set positions; remove their probe-set copies
    poison_scores[exp.train_probes_idx] = 0.
    poison_scores[probe_new_idx] = 1.1

    threshold = 1 - args.badloss_reject_frac
    exp.wandb_prefix = f"cleaned_thresh_{threshold}_"
    clean_indices = np.where(poison_scores <= threshold)[0]
    print(f"!! [Dataset cleansing] Total examples: {len(poison_scores)} / # clean indices: {len(clean_indices)}")
    discarded_indices = [i for i in range(len(poison_scores)) if i not in clean_indices and i not in missing_vals_idx]
    discarded_identities = Counter(dataset_probe_identity[i] for i in discarded_indices)
    print("!! Discarded example identities:", discarded_identities)
    exp.record_detection(num_removed=len(discarded_indices), removed_identities=dict(discarded_identities))

    retrain_loader = exp.loader(clean_indices)
    clean_model = exp.new_model()
    criterion, optimizer, lr_scheduler = exp.new_optimizer(clean_model, exp.num_epochs)
    output_checkpoint_dir = os.path.join(exp.output_dir, "model_ft")
    os.makedirs(output_checkpoint_dir, exist_ok=True)
    output_checkpoint = os.path.join(output_checkpoint_dir, f"model_ft_cleaned_thresh_{threshold:.2f}.pth")
    print("Selected output checkpoint:", output_checkpoint)
    if not os.path.exists(output_checkpoint):
        print("!! Output checkpoint not found. Training model from scratch...")
        for _ in range(exp.num_epochs):
            train(clean_model, exp.device, retrain_loader, optimizer, criterion)
            lr_scheduler.step()
        torch.save(clean_model.state_dict(), output_checkpoint)
        print("!! Final checkpoint written to file:", output_checkpoint)
    else:
        print("!! Loading model from pretrained checkpoint:", output_checkpoint)
        clean_model.load_state_dict(torch.load(output_checkpoint, map_location=exp.device))

    asr = exp.evaluate_asr(clean_model, criterion)
    test_stats, _ = test(clean_model, exp.device, criterion, exp.test_idx_loader, exp.distributed, exp.rank)
    exp.record_model_metrics("retrained", test_stats, asr)
