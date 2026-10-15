"""Spectral Signatures (Tran et al., 2018).

For every class, center the penultimate representations, project onto the top singular vector, and remove the
1.5 * eps fraction of examples with the largest squared projection; then retrain. With --ss_oracle, attack target
classes instead lose exactly as many examples as they contain poisons.
"""
import numpy as np
import torch


def get_indices_for_eps(eps_thresh, num_classes, taus, cls_idx, indices, oracle_counts=None):
    """oracle_counts: optional class -> number of examples to remove, overriding eps for those classes."""
    rejected_indices = []
    for cls in range(num_classes):
        tau = taus[cls]
        num_to_remove = int(len(tau) * eps_thresh * 1.5)
        if oracle_counts is not None and cls in oracle_counts:
            num_to_remove = oracle_counts[cls]
        if num_to_remove == 0:  # (a [-0:] slice would select the whole class)
            continue
        rejected_indices.append(cls_idx[cls][tau.argsort()[-num_to_remove:].cpu()])
    if not rejected_indices:
        return np.array([], dtype=int)
    rejected_indices = torch.hstack(rejected_indices).unique()
    return indices[rejected_indices.cpu()].numpy()


def run(exp):
    args = exp.args
    exp.pretrain()
    exp.report_attacked_model()
    exp.wandb_prefix = "retraining_"

    activations, indices, classes, predictions = exp.last_layer_activations(exp.model, exp.new_idx_loader_wo_aug)
    taus, cls_idx = [], []
    for cls in range(exp.num_classes):
        cls_indices = (classes == cls).nonzero()[:, 0]
        m = activations[cls_indices] - activations[cls_indices].mean(axis=0)
        u, s, v = m.svd()
        taus.append(m.matmul(v[:, 0]) ** 2)
        cls_idx.append(cls_indices)

    auc_idx = [get_indices_for_eps(t, exp.num_classes, taus, cls_idx, indices)
               for t in np.geomspace(0.01, 0.5, num=50)]
    exp.report_detection_auc("SS", auc_idx)

    oracle_counts = None
    if args.ss_oracle:  # The AUC above doesn't depend on the oracle (it sweeps the removal fraction)
        is_poison = exp.is_poison()
        oracle_counts = {cls: int(is_poison[indices[cls_idx[cls]].cpu().numpy()].sum()) for cls in exp.target_classes()}
        print("SS oracle removal counts:", oracle_counts)
        exp.record_detection(oracle_counts=oracle_counts)
    identified_indices = get_indices_for_eps(args.ss_eps, exp.num_classes, taus, cls_idx, indices, oracle_counts)
    exp.report_detection(identified_indices)
    exp.retrain(identified_indices)
    print("Done with ss")
