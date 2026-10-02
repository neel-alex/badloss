from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import sklearn.metrics


def get_confusion_stats(identified_indices, valid_idx, dataset_probe_identity: Sequence[str],
                        verbose: bool = True, per_class_out: Optional[Dict] = None) -> Tuple[float, float]:
    """FPR and TPR of a detector over the combined training set.

    identified_indices: indices (into the combined training set) flagged as poisoned.
    valid_idx: indices of the unmodified training examples (all clean).
    dataset_probe_identity: identity of every combined-training-set index ('train' for unmodified examples).
    per_class_out: if given, filled with the per-identity detection rate.
    """
    false_pos = np.intersect1d(identified_indices, valid_idx)
    true_pos = np.array([])
    counter = {k: 0 for k in set(dataset_probe_identity) - {'train'}}
    results = {k: 0 for k in set(dataset_probe_identity) - {'train'}}

    for i, id in enumerate(dataset_probe_identity):
        if id == 'train':
            continue
        if i in identified_indices:
            results[id] += 1
            if id == 'clean' or id == "clean_val":
                false_pos = np.append(false_pos, i)
            else:
                true_pos = np.append(true_pos, i)
        counter[id] += 1

    num_attacks = sum(v for (k, v) in counter.items() if 'clean' not in k)
    num_clean = len(valid_idx) + sum(v for (k, v) in counter.items() if 'clean' in k)

    false_positive = len(false_pos)
    true_positive = len(true_pos)
    true_negative = num_clean - false_positive
    false_negative = num_attacks - true_positive

    per_class = {attack: results[attack] / counter[attack]
                 for attack in set(dataset_probe_identity) - {'train'} if 'clean' not in attack}

    if verbose:
        print(f"FPR: {false_positive / (false_positive + true_negative)}")
        print(f"FNR: {false_negative / (false_negative + true_positive)}")
        print("Per-attack detection rates...")
        for k in per_class:
            print(f"Detection rate ({k}): {per_class[k]}")

    if per_class_out is not None:
        per_class_out.update(per_class)
    false_positive_rate = false_positive / (false_positive + true_negative)
    true_positive_rate = true_positive / (true_positive + false_negative)
    return false_positive_rate, true_positive_rate


def get_auc(idx_list, valid_idx, dataset_probe_identity: Sequence[str]) -> float:
    """Area under the ROC curve traced by the detector's flagged sets at a sweep of thresholds."""
    fprs, tprs = [], []

    for identified_indices in idx_list:
        fpr, tpr = get_confusion_stats(identified_indices, valid_idx, dataset_probe_identity, verbose=False)
        fprs.append(fpr)
        tprs.append(tpr)

    fprs, tprs = zip(*sorted(zip(fprs, tprs)))
    return sklearn.metrics.auc(fprs, tprs)
