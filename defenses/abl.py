"""Anti-Backdoor Learning (Li et al., 2021), adapted to filter the training set and retrain.

Train briefly with loss flooding, then flag the examples with the lowest loss. As in the paper, the lowest 15%
are removed and the model is retrained from scratch (ABL's unlearning step was unstable in testing).
"""
import numpy as np
from tqdm import tqdm

from torch_utils import train, test


def get_indices_from_losses(thresh, num_train, loss_idx, ex_idx):
    """The thresh * num_train lowest-loss examples (as combined-training-set indices)."""
    return np.array([int(ex_idx[i]) for i in loss_idx[:int(num_train * thresh)]])


def run(exp):
    args = exp.args
    exp.wandb_prefix = "retraining_"
    model = exp.new_model()

    print("!! Performing initial pretraining with all examples (using loss flooding)...")
    criterion, optimizer, _ = exp.new_optimizer(model, args.abl_pretrain_epochs)
    for epoch in tqdm(range(args.abl_pretrain_epochs)):
        train(model, exp.device, exp.new_idx_loader, optimizer, criterion, flooding_threshold=args.abl_flooding)
        if epoch % 5 == 4:
            exp.evaluate(model, criterion)

    # Per-example losses over the training set (NB: computed on the augmented loader)
    _, pred_output_dict = test(model, exp.device, criterion, exp.new_idx_loader, exp.distributed, exp.rank,
                               log_predictions=True)
    loss_idx = np.argsort(pred_output_dict["loss"])  # Ascending
    ex_idx = pred_output_dict["ex_idx"]
    num_train = len(exp.train_set)

    auc_idx = [get_indices_from_losses(t, num_train, loss_idx, ex_idx) for t in np.geomspace(0.001, 0.5, num=50)]
    exp.report_detection_auc("ABL", auc_idx)

    identified_indices = get_indices_from_losses(args.abl_remove_frac, num_train, loss_idx, ex_idx)
    exp.report_detection(identified_indices)
    exp.retrain(identified_indices)
