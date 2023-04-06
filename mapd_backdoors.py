#!/usr/bin/env python
# coding: utf-8

# ## Inserting probes into the model for inspecting model phase

# In[ ]:

##

import copy
import itertools
import natsort
import os
import pickle
import shutil
import sys
import warnings
import random
from tqdm import tqdm

import numpy as np
import cv2
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sklearn.neighbors
from sklearn.metrics import confusion_matrix
from catalyst.data import DistributedSamplerWrapper


import dist_utils
from dataset_utils import get_settings_for_dataset, make_probe_dataset, make_index_dataset, combine_dataset
from plot_utils import plot_probe_examples, plot_probe_ex, some_plot, \
    some_other_plot, make_normalizers, yet_another_plot, one_more_plot
from plot_utils import num_queue_plots
from backdoors import make_probes
from torch_utils import make_model, train, test, test_tensor


# Set random seed
seed = 3
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed=seed)
random.seed(seed)

# Plotting config
include_plot_title = False
font_size = 16


dataset_choices = ["mnist", "cifar10", "cifar100", "gtsrb", "imagenet"]

if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} <Dataset: {'/'.join(dataset_choices)}>")
    exit()

dataset = sys.argv[1]
assert dataset in dataset_choices


# Essential config
log_predictions = True
distributed = True if dataset == "imagenet" else False
num_train_probes = 250
num_val_probes = 250
use_val_probes_for_training = True
num_example_probes = num_train_probes + num_val_probes
random_backdoor_alpha = 0.1
experiment_output_dir = f"./backdoor_exp05_{dataset}_alpha_{random_backdoor_alpha}"
num_workers = 8
surface_examples = False
aux_loss_lambda = 1.0  # Based on the experiments with center loss
feat_dim = 2048  # Feature dimensions for ResNet-50

print("Dataset:", dataset)
print("Distributed training:", distributed)


# Initialize the distributed environment
gpu = 0
world_size = 1
distributed = distributed or int(os.getenv('WORLD_SIZE', 1)) > 1
rank = int(os.getenv('RANK', 0))
local_rank = 0

if "SLURM_NNODES" in os.environ:
    local_rank = rank % torch.cuda.device_count()
    print(f"SLURM tasks/nodes: {os.getenv('SLURM_NTASKS', 1)}/{os.getenv('SLURM_NNODES', 1)}")
elif "WORLD_SIZE" in os.environ:
    local_rank = int(os.getenv('LOCAL_RANK', 0))

if distributed:
    gpu = local_rank
    torch.cuda.set_device(gpu)
    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    world_size = torch.distributed.get_world_size()
    assert int(os.getenv('WORLD_SIZE', 1)) == world_size
    print(f"Initializing the environment with {world_size} processes | Current process rank: {local_rank}")

main_proc = dist_utils.is_main_proc(local_rank, shared_fs=True)
print("Is main proc?", main_proc)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


setup_for_distributed(main_proc)
warnings.filterwarnings("ignore", "Warning: Leaking Caffe2 thread-pool after fork. (function pthreadpool)", UserWarning)


recompute_results = False
if main_proc:
    if recompute_results:
        if os.path.exists(experiment_output_dir):
            shutil.rmtree(experiment_output_dir)
        os.makedirs(experiment_output_dir)
    else:
        if not os.path.exists(experiment_output_dir):
            os.makedirs(experiment_output_dir)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Current device:", device)


# Shunting off this logic to another file -- TODO select file location arg?
(img_size, train_transform, test_transform, no_transform,
 data_dir, train_set, train_set_wo_aug, test_set) = get_settings_for_dataset(dataset)


print(dataset, len(train_set), len(test_set))

# ## Setup probes

num_classes = len(train_set.classes)
if dataset in ["mnist", "cifar10"]:
    assert num_classes == 10
elif dataset == "cifar100":
    assert num_classes == 100
elif dataset == "gtsrb":
    assert num_classes == 43
else:
    assert num_classes == 1000
print(dataset, num_classes)


attack_types = ["reversed", "single_pix", "reversed_single_pix", "random", "warped"]
probes = make_probes(num_classes, train_set, train_set_wo_aug, num_example_probes, attack_types,
                     random_backdoor_alpha, experiment_output_dir, main_proc, img_size, device)


plot_probe_examples(probes, dataset, train_set, attack_types, rank, experiment_output_dir)


# Hyperparameters
if dataset == "mnist":
    num_epochs = 25
    batch_size = 256
elif "cifar" in dataset:
    num_epochs = 150
    batch_size = 128
else:
    assert dataset == "imagenet" or dataset == "gtsrb"
    num_epochs = 100
    optimizer_batch_size = 256
    batch_size = 256
    if distributed:
        assert batch_size % world_size == 0
        batch_size = batch_size // world_size
        print(f"Optimizer batch size: {optimizer_batch_size} / World size: {world_size} / Local batch size: {batch_size}")
lr = 0.1
momentum = 0.9
wd = 0.0001


(probe_dataset_standard, val_probe_dataset_standard, val_probes,
 train_indices, probe_identity, val_probe_identity, discarded_idx) = \
    make_probe_dataset(probes, train_set, test_set, dataset, batch_size, num_example_probes, num_train_probes,
                       num_val_probes, attack_types, distributed, num_workers, experiment_output_dir, device)


model = make_model(dataset, num_classes, device)
print(model)

# Convert to a distributed model
model = dist_utils.convert_to_distributed(model, local_rank, sync_bn=True)

criterion = torch.nn.CrossEntropyLoss(reduction='none').to(device)  # reduction='mean' by default
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
scaler = torch.cuda.amp.GradScaler()


comb_train_set, comb_train_indices, dataset_probe_identity = \
    combine_dataset(train_set, train_indices, probe_dataset_standard, val_probe_dataset_standard,
                    probe_identity, val_probe_identity, use_val_probes_for_training)

new_idx_loader, new_idx_loader_wo_aug, test_idx_loader = \
    make_index_dataset(comb_train_set, comb_train_indices, test_set,
                       no_transform, batch_size, distributed, num_workers)


model_file = os.path.join(experiment_output_dir, f"models_{dataset}", f"model_{dataset}_dynamics.pth")
data_file = os.path.join(experiment_output_dir, f"stats_{dataset}_dynamics.pkl")
data_statistics_file = os.path.join(experiment_output_dir, f"stats_{dataset}_data_statistics.pkl")


# In[ ]:


model_dir = os.path.split(model_file)[0]
if main_proc and not os.path.exists(model_dir):
    os.mkdir(model_dir)


# In[ ]:


ref_probe_classes = ["backdoor", "clean"]
label_map_dict = {"backdoor": "Backdoor", "backdoor_val": "Backdoor [Val]",
                  "backdoor_random_val": "Backdoor (random) [Val]",
                  "backdoor_reversed_val": "Backdoor (reversed) [Val]",
                  "backdoor_single_pix_val": "Backdoor (single pixel) [Val]",
                  "backdoor_reversed_single_pix_val": "Backdoor (reversed single pixel) [Val]",
                  "backdoor_warped": "Backdoor (warped)", "backdoor_warped_val": "Backdoor (warped) [Val]",
                  "clean": "Clean", "clean_val": "Clean [Val]", "train": "Train", "test": "Test"}


# In[ ]:


if not os.path.exists(model_file):
    statistics = {"train": [], "test": []}
    statistics.update({k: [] for k in ref_probe_classes})
    statistics.update({f"{k}_val": [] for k in ref_probe_classes})  # Add keys for validation probes
    statistics.update({f"backdoor_{k}_val": [] for k in attack_types})  # Adding additional validation keys
    inv_probe_map = {i: v for i, v in enumerate(ref_probe_classes)}
    
    surface_epoch = 5
    
    predictions = {}
    
    for epoch in range(num_epochs):
        output_dict = train(model, device, new_idx_loader, optimizer, criterion, scaler)
        
        # Collect test set statistics
        print("Stats for epoch #", epoch+1)
        if log_predictions:
            # Don't use train_idx_loader here -- also assumes that probes are include for later evaluation
            train_stats, train_preds = test(model, device, criterion, new_idx_loader_wo_aug, distributed, rank, set_name="Train", log_predictions=log_predictions)
        
        test_stats, test_preds = test(model, device, criterion, test_idx_loader, distributed, rank, log_predictions=log_predictions)
        
        # Collect probe statistics
        backdoor_stats, backdoor_preds = test_tensor(model, device, criterion, probes["backdoor"], probes["backdoor_labels"], msg="Backdoor probe", log_predictions=log_predictions)
        val_backdoor_stats, val_backdoor_preds = test_tensor(model, device, criterion, val_probes["backdoor"], val_probes["backdoor_labels"], msg="Backdoor probe (val)", log_predictions=log_predictions)
        clean_stats, clean_preds = test_tensor(model, device, criterion, probes["clean"], probes["clean_labels"], msg="Clean probe", log_predictions=log_predictions)
        val_clean_stats, val_clean_preds = test_tensor(model, device, criterion, val_probes["clean"], val_probes["clean_labels"], msg="Clean probe (val)", log_predictions=log_predictions)
        
        val_backdoor_reversed_stats, val_backdoor_reversed_preds = test_tensor(model, device, criterion, val_probes["backdoor_reversed"], val_probes["backdoor_reversed_labels"],
                                                                               msg="Backdoor reversed probe (val)", log_predictions=log_predictions)
        val_backdoor_single_pix_stats, val_backdoor_single_pix_preds = test_tensor(model, device, criterion, val_probes["backdoor_single_pix"], val_probes["backdoor_single_pix_labels"],
                                                                                   msg="Backdoor single pixel probe (val)", log_predictions=log_predictions)
        val_backdoor_reversed_single_pix_stats, val_backdoor_reversed_single_pix_preds = test_tensor(model, device, criterion, val_probes["backdoor_reversed_single_pix"],
                                                                                                     val_probes["backdoor_reversed_single_pix_labels"],
                                                                                                     msg="Backdoor single pixel reversed probe (val)", log_predictions=log_predictions)
        val_backdoor_random_stats, val_backdoor_random_preds = test_tensor(model, device, criterion, val_probes["backdoor_random"], val_probes["backdoor_random_labels"],
                                                                           msg="Backdoor random (val)", log_predictions=log_predictions)
        val_backdoor_warped_stats, val_backdoor_warped_preds = test_tensor(model, device, criterion, val_probes["backdoor_warped"], val_probes["backdoor_warped_labels"],
                                                                           msg="Backdoor warped (val)", log_predictions=log_predictions)
        
        if log_predictions:
            statistics["train"].append(train_stats)
            
            # Add predictions from all the different sets / probes
            predictions[epoch] = {}  # Dict of dict
            predictions[epoch]["train"] = train_preds
            predictions[epoch]["test"] = test_preds
            predictions[epoch]["clean"] = backdoor_preds
            predictions[epoch]["clean_val"] = val_clean_preds
            predictions[epoch]["backdoor"] = backdoor_preds
            predictions[epoch]["backdoor_val"] = val_backdoor_preds
            predictions[epoch]["backdoor_reversed_val"] = val_backdoor_reversed_preds
            predictions[epoch]["backdoor_single_pix_val"] = val_backdoor_single_pix_preds
            predictions[epoch]["backdoor_reversed_single_pix_val"] = val_backdoor_reversed_single_pix_preds
            predictions[epoch]["backdoor_random_val"] = val_backdoor_random_preds
            predictions[epoch]["backdoor_warped_val"] = val_backdoor_warped_preds
        
        statistics["test"].append(test_stats)
        statistics["clean"].append(clean_stats)
        statistics["clean_val"].append(val_clean_stats)
        statistics["backdoor"].append(backdoor_stats)
        statistics["backdoor_val"].append(val_backdoor_stats)
        statistics["backdoor_reversed_val"].append(val_backdoor_reversed_stats)
        statistics["backdoor_single_pix_val"].append(val_backdoor_single_pix_stats)
        statistics["backdoor_reversed_single_pix_val"].append(val_backdoor_reversed_single_pix_stats)
        statistics["backdoor_random_val"].append(val_backdoor_random_stats)
        statistics["backdoor_warped_val"].append(val_backdoor_warped_stats)
        
        if lr_scheduler is not None:
            lr_scheduler.step()
        
        if main_proc:
            # Save the model
            model_file_base, model_file_ext = os.path.splitext(model_file)
            current_model_file = f"{model_file_base}_ep_{epoch}{model_file_ext}"
            torch.save(model.state_dict(), current_model_file)
        
        # Close all figures
        plt.close('all')
    
    if log_predictions:
        statistics["predictions"] = predictions

    if main_proc:
        # Save the model
        torch.save(model.state_dict(), model_file)

        # Save the final data
        with open(data_file, "wb") as f:
            pickle.dump(statistics, f, protocol=pickle.HIGHEST_PROTOCOL)
else:
    assert os.path.exists(data_file)
    print("Data files already found. Loading data from saved checkpoints...")
    
    model.load_state_dict(torch.load(model_file, map_location=device))
    with open(data_file, "rb") as f:
        statistics = pickle.load(f)


print("Final test accuracy:", statistics["test"][-1])


some_plot(statistics, log_predictions, label_map_dict, include_plot_title, dataset, main_proc, experiment_output_dir)
some_other_plot(statistics, log_predictions, label_map_dict, include_plot_title, dataset, main_proc, experiment_output_dir)


if not log_predictions:
    print("Can't compute other statistics without the model predictions...")
    exit()


# ### Learning dynamics per example


unique_probe_identity = np.unique(dataset_probe_identity)
print("Unique dataset probe identity:", unique_probe_identity)


if not os.path.exists(data_statistics_file):
    sorted_ex_list = []
    num_total_vals = len(comb_train_set)  # 50000 + 600

    print("Computing sorted prediction and target list...")
    for k in tqdm(statistics["predictions"].keys()):  # Iterate over epochs
        assert "train" in statistics["predictions"][k], statistics["predictions"][k].keys()
        out_dict = statistics["predictions"][k]["train"]
        ex_idx = out_dict["ex_idx"]
        preds = out_dict["preds"]
        targets = out_dict["targets"]
        num_extra_indices = len(ex_idx[np.nonzero(ex_idx >= len(train_set))])
        sorted_preds = np.ones_like(preds, shape=(num_total_vals,)) * -1
        sorted_targets = np.ones_like(targets, shape=(num_total_vals,)) * -1
        unused_idx = list(range(num_total_vals))
        unused_idx = [x for x in unused_idx if x not in ex_idx]
        for i in range(len(preds)):
            current_ex_idx = ex_idx[i]
            sorted_preds[current_ex_idx] = preds[i]
            sorted_targets[current_ex_idx] = targets[i]
        assert np.sum(sorted_preds == -1) == len(unused_idx)
        sorted_ex_list.append((sorted_preds, sorted_targets, unused_idx))


    # In[ ]:


    epoch_learned = np.ones((num_total_vals,), dtype=np.int64) * -1
    epoch_first_learned = np.ones((num_total_vals,), dtype=np.int64) * -1

    print("Computing first-learned statistics...")
    for epoch in tqdm(range(len(sorted_ex_list))):  # Iterate over the epochs
        preds, targets, _ = sorted_ex_list[epoch]
        
        # Update current epoch learned
        preds[preds == -1] = -2  # Set the preds to be -2 just to make sure the targets and predictions don't match
        correct_examples_current = preds == targets
        previously_correct_ex = epoch_learned != -1
        
        mark_unlearned = np.logical_and(previously_correct_ex, np.logical_not(correct_examples_current))
        mark_learned = np.logical_and(np.logical_not(previously_correct_ex), correct_examples_current)
        epoch_learned[mark_unlearned] = -1
        epoch_learned[mark_learned] = epoch
        
        # Update example learned for the first time
        previously_correct_ex = epoch_first_learned != -1
        mark_learned = np.logical_and(np.logical_not(previously_correct_ex), correct_examples_current)
        epoch_first_learned[mark_learned] = epoch
        
        print(f"Epoch: {epoch} \t Previously learned examples: {np.sum(previously_correct_ex)} \t Newly learned examples: {np.sum(mark_learned)} \t Examples marked as unlearned: {np.sum(mark_unlearned)} \t Total new learned examples: {np.sum(epoch_learned != -1)} \t Total learned examples at any time: {np.sum(epoch_first_learned != -1)}")


    # In[ ]:


    stats = {k: 0 for k in unique_probe_identity}
    epoch_cumulative_scores = {k: [] for k in unique_probe_identity}

    print("Computing cumulative statistics...")
    for epoch in tqdm(range(num_epochs)):
        examples_learned_at_epoch = epoch_learned == epoch
        learned_ex_idx = np.nonzero(examples_learned_at_epoch)[0]
        for i in learned_ex_idx:
            k = dataset_probe_identity[i]
            stats[k] += 1
        for k in unique_probe_identity:
            epoch_cumulative_scores[k].append(stats[k])

    print("Statistics:", stats)
    print("Cumulative stats:", epoch_cumulative_scores)
    total_examples_learned = 0
    for k in stats:
        total_examples_learned += stats[k]
    print("Total examples learned in the end:", total_examples_learned)
    assert total_examples_learned == (len(epoch_learned) - int(np.sum(epoch_learned == -1)))


    # First learned stats
    stats_first_learned = {k: 0 for k in unique_probe_identity}
    epoch_cumulative_scores_first_learned = {k: [] for k in unique_probe_identity}

    print("Computing first-learned statistics...")
    for epoch in range(num_epochs):
        examples_learned_at_epoch = epoch_first_learned == epoch
        learned_ex_idx = np.nonzero(examples_learned_at_epoch)[0]
        for i in learned_ex_idx:
            k = dataset_probe_identity[i]
            stats_first_learned[k] += 1
        for k in unique_probe_identity:
            epoch_cumulative_scores_first_learned[k].append(stats_first_learned[k])

    print("Statistics:", stats_first_learned)
    print("Cumulative stats:", epoch_cumulative_scores_first_learned)
    total_examples_learned = 0
    for k in stats:
        total_examples_learned += stats_first_learned[k]
    print("Total examples learned at any point during training:", total_examples_learned)
    assert total_examples_learned == (len(epoch_first_learned) - int(np.sum(epoch_first_learned == -1)))

    if main_proc:
        # Save the final statistics
        with open(data_statistics_file, "wb") as f:
            final_statistics = [sorted_ex_list, epoch_learned, epoch_first_learned, stats, epoch_cumulative_scores, stats_first_learned, epoch_cumulative_scores_first_learned]
            pickle.dump(final_statistics, f, protocol=pickle.HIGHEST_PROTOCOL)
else:
    assert os.path.exists(data_statistics_file)
    print("Data files already found. Loading data statistics from file:", data_statistics_file)
    
    with open(data_statistics_file, "rb") as f:
        final_statistics = pickle.load(f)
        sorted_ex_list, epoch_learned, epoch_first_learned, stats, epoch_cumulative_scores, stats_first_learned, epoch_cumulative_scores_first_learned = final_statistics


normalizers = make_normalizers(num_train_probes, train_set, discarded_idx, unique_probe_identity)
yet_another_plot(statistics, normalizers, epoch_cumulative_scores, epoch_cumulative_scores_first_learned,
                 label_map_dict, include_plot_title, dataset, main_proc, experiment_output_dir)

# ### Loss distribution plots

# In[ ]:


# List of example_idx at different epochs i.e. [epoch_1_loss_vals, ...., epoch_n_loss_vals]
ex_idx = [statistics["predictions"][i]["train"]["ex_idx"] for i in range(len(statistics["predictions"]))]
loss_values = [statistics["predictions"][i]["train"]["loss"] for i in range(len(statistics["predictions"]))]
print(len(ex_idx), len(loss_values))


# In[ ]:


sorted_losses_all = []
assert len(dataset_probe_identity) == len(comb_train_set)

print("Computing the sorted loss list...")
for i in range(len(ex_idx)):  # Iterate over the epochs
    current_ex_idx = ex_idx[i]
    current_loss_vals = loss_values[i]
    assert len(current_ex_idx) == len(current_loss_vals), f"{len(current_ex_idx)} != {len(current_loss_vals)}"
    current_sorted_loss_vals = [None for _ in range(len(dataset_probe_identity))]  # Includes both the training set as well as the probes i.e. len(comb_train_set)
    for j, k in enumerate(current_ex_idx):
        current_sorted_loss_vals[k] = current_loss_vals[j]
    sorted_losses_all.append(current_sorted_loss_vals)


class_names = list(np.unique(dataset_probe_identity))
print(class_names)


one_more_plot(sorted_losses_all, class_names, label_map_dict, dataset_probe_identity,
                  dataset, main_proc, experiment_output_dir)

loss_dynamics_output_dir = os.path.join(experiment_output_dir, "loss_distribution")
violin_loss_dynamics_output_dir = os.path.join(experiment_output_dir, "loss_distribution_violin")
if main_proc and not os.path.exists(loss_dynamics_output_dir):
    os.mkdir(loss_dynamics_output_dir)
if main_proc and not os.path.exists(violin_loss_dynamics_output_dir):
    os.mkdir(violin_loss_dynamics_output_dir)



for epoch in range(0, len(sorted_losses_all), 5):
    fig, ax = plt.subplots(1, 1, figsize=(5, 6))
    labels = list(range(1, len(sorted_losses_all)+1))
    color_list = ['tab:red', 'tab:blue', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:cyan', 'tab:olive', 'tab:gray']
    plot_points = False

    handles = []
    legend_label = []
    data = []
    iterator = 0
    
    rej_classes = []
    
    for i, cls in enumerate(class_names):
        if cls in rej_classes:
            print(f"Ignoring class {cls} at index {i}")
            continue
        print("Class:", cls)
        color = color_list[iterator % len(color_list)]
        patch = mpatches.Patch(color=color)
        handles.append(patch)
        # legend_label.append(cls.replace("_", " ").title())
        legend_label.append(label_map_dict[cls])

        data = [[] for _ in range(len(class_names)-len(rej_classes))]
        current_losses = [float(sorted_losses_all[epoch][i]) for i in range(len(sorted_losses_all[epoch])) if str(dataset_probe_identity[i]) == cls and sorted_losses_all[epoch][i] is not None]
        data[iterator] = current_losses
        
        parts = ax.boxplot(data, notch=True, patch_artist=True, showfliers=False,
                   boxprops=dict(facecolor=color, color=color, alpha=1.0),
                   capprops=dict(color=color),
                   whiskerprops=dict(color=color),
                   flierprops=dict(color=color, markeredgecolor=color),
                   medianprops=dict(color=color))
        
        iterator += 1

    # ax.legend(handles, legend_label, prop={'size': font_size})
    plt.ylabel("Loss values", fontsize=font_size)
    ax.set_xticks(range(1, len(legend_label)+1))
    ax.set_xticklabels(legend_label, fontsize=font_size)
    plt.xticks(rotation=90)
    plt.yticks(fontsize=font_size-2)
    plt.ylim(0., 14.)
    
    plt.tight_layout()
    output_file = os.path.join(loss_dynamics_output_dir, f"loss_dist_ep_{epoch}_{dataset}.png")
    if main_proc and output_file is not None:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close('all')


# In[ ]:


for epoch in range(0, len(sorted_losses_all), 5):
    fig, ax = plt.subplots(1, 1, figsize=(5, 6))
    labels = list(range(1, len(sorted_losses_all)+1))
    color_list = ['tab:red', 'tab:blue', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:cyan', 'tab:olive', 'tab:gray']
    plot_points = False

    handles = []
    legend_label = []
    data = []
    iterator = 0
    
    print("Rejected classes:", rej_classes)
    
    for i, cls in enumerate(class_names):
        if cls in rej_classes:
            print(f"Ignoring class {cls} at index {i}")
            continue
        print("Class:", cls)
        color = color_list[iterator % len(color_list)]
        patch = mpatches.Patch(color=color)
        handles.append(patch)
        # legend_label.append(cls.replace("_", " ").title())
        legend_label.append(label_map_dict[cls])

        data = [[float('nan'), float('nan')] for _ in range(len(class_names)-len(rej_classes))]
        current_losses = [float(sorted_losses_all[epoch][i]) for i in range(len(sorted_losses_all[epoch])) if str(dataset_probe_identity[i]) == cls and sorted_losses_all[epoch][i] is not None]
        data[iterator] = current_losses
        
        parts = ax.violinplot(data, showmeans=False, showmedians=True, showextrema=False, widths=0.8)
        for part_name in ['cbars','cmins','cmaxes','cmeans','cmedians']:
            if part_name in parts:
                pc = parts[part_name]
                pc.set_edgecolor(color)
                pc.set_linewidth(1)
        for pc in parts['bodies']:
            pc.set_facecolor(color)
        
        # Plot the points
        num_points = 250
        include_points = True
        if include_points:
            ax.scatter([iterator+1 for _ in range(num_points)], np.random.choice(data[iterator], num_points), alpha=0.1, color=color)
        
        iterator += 1

    # ax.legend(handles, legend_label, prop={'size': font_size})
    plt.ylabel("Loss values", fontsize=font_size)
    ax.set_xticks(range(1, len(legend_label)+1))
    ax.set_xticklabels(legend_label, fontsize=font_size)
    plt.xticks(rotation=90)
    plt.yticks(fontsize=font_size-2)
    plt.ylim(0., 14.)

    plt.tight_layout()
    output_file = os.path.join(violin_loss_dynamics_output_dir, f"loss_dist_violin_ep_{epoch}_{dataset}.png")
    if main_proc and output_file is not None:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close('all')


# In[ ]:


def visualize_loss_trajectories(val_included=False, clf=None, output_file=None):
    current_class_names = [x for x in class_names if x not in ["train", "train_noisy", "train_non_noisy"]]
    if not val_included:
        current_class_names = [x for x in current_class_names if not x.endswith("_val")]
    print("Selected class names:", current_class_names)
    
    num_colors = len(current_class_names)
    if num_colors > 9:
        cm = plt.get_cmap('hsv')
        color_list = [cm(1.*i/len(current_class_names)) for i in range(len(current_class_names))]
    elif num_colors > 4:
        color_list = ["tab:green", "tab:blue", "tab:purple", "tab:orange", "tab:red", "tab:pink", "tab:olive", "tab:brown", "tab:cyan"]
    else:
        color_list = ["tab:green", "tab:blue", "tab:purple", "tab:orange"]
    assert num_colors <= len(color_list), f"{num_colors} <= {len(color_list)}"
    num_trajectories = 250
    font_size = 18

    fig, ax = plt.subplots(1, 1, figsize=(20, 8))

    handles = []
    legend_label = []

    iterator = 0
    traj_list = []
    for i, cls in enumerate(current_class_names):
        # if "val" in cls or "train" in cls:
        #     continue
        color = color_list[iterator]
        patch = mpatches.Patch(color=color)
        handles.append(patch)
        # legend_label.append(cls.title().replace("_", " "))
        legend_label.append(label_map_dict[cls])
        
        relevant_idx = [i for i in range(len(dataset_probe_identity)) if dataset_probe_identity[i] == cls]
        print(f"Class: {cls} / # relevant idx: {len(relevant_idx)}")

        x_axis = list(range(len(sorted_losses_all)))
        all_trajs = []
        for j in range(num_trajectories):
            trajectory = [float(sorted_losses_all[epoch][relevant_idx[j]]) for epoch in range(len(sorted_losses_all))]
            plt.plot(x_axis, trajectory, color=color_list[iterator], alpha=0.05)
            all_trajs.append(trajectory)
        traj_list += all_trajs
        
        if clf is None:
            # Plot the trajectory mean
            mean_traj = np.array(all_trajs).mean(axis=0)
            plt.plot(x_axis, mean_traj, color=color_list[iterator], alpha=0.9, linewidth=5.)
        iterator += 1
    
    if clf is not None:
        num_clusters = len(clf.cluster_centers_)
        cm = plt.get_cmap('viridis')
        new_color_list = [cm(1.*i/num_clusters) for i in range(num_clusters)]
        
        for i in range(num_clusters):
            # Plot the cluster center
            color = new_color_list[i]
            cluster_center = clf.cluster_centers_[i]
            plt.plot(x_axis, cluster_center, color=color, alpha=0.9, linewidth=5.)
            
            # Add the color to the legend
            patch = mpatches.Patch(color=color)
            handles.append(patch)
            legend_label.append(f"Cluster # {i+1}")
    
    ax.legend(handles, legend_label, prop={'size': font_size})
    plt.ylabel("Loss values", fontsize=font_size)
    plt.xlabel("Epochs", fontsize=font_size)
    max_val = np.percentile(traj_list, 99)
    plt.ylim(0., max_val)
    plt.xlim(0., len(x_axis)-1)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)

    plt.tight_layout()
    if output_file is None:
        output_file = os.path.join(experiment_output_dir, f"loss_trajectories_{dataset}{'_val' if val_included else ''}.png")
    if main_proc and output_file is not None:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close('all')


# In[ ]:


for val_included in [True, False]:
    visualize_loss_trajectories(val_included)


# In[ ]:


# Convert the data into a complete trajectory dataset
print("Converting trajectories to dataset...")
dataset = {}
for i, cls in enumerate(class_names):
    relevant_idx = [i for i in range(len(dataset_probe_identity)) if dataset_probe_identity[i] == cls]
    print(f"Class: {cls} / # relevant idx: {len(relevant_idx)}")
    
    dataset[cls] = []
    empty_idx = []
    for j in range(len(relevant_idx)):
        if sorted_losses_all[0][relevant_idx[j]] is None:  # The whole trajectory should be none since these examples are used in probes
            assert all([sorted_losses_all[epoch][relevant_idx[j]] is None for epoch in range(len(sorted_losses_all))])
            empty_idx.append(relevant_idx[j])
            continue
        trajectory = [float(sorted_losses_all[epoch][relevant_idx[j]]) for epoch in range(len(sorted_losses_all))]
        dataset[cls].append(trajectory)
    assert len(dataset[cls]) == len(relevant_idx) - len(empty_idx)
    if len(empty_idx) > 0:
        print("Number of empty trajectories:", len(empty_idx))

print("Total number of keys found:", dataset.keys(), {k: len(dataset[k]) for k in dataset.keys()})
trajectory_dataset_file = os.path.join(experiment_output_dir, f"loss_trajectories.pkl")
with open(trajectory_dataset_file, "wb") as f:
    pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
print("Trajectory dataset written to file:", trajectory_dataset_file)


# In[ ]:


print("Converting trajectories to numpy dataset...")
class_names = natsort.natsorted(list(dataset.keys()))
print("Class names:", class_names)

main_classes = [x for x in class_names if not x.endswith("_val") and x != "train"]
print(main_classes)

class2idx = {k: i for i, k in enumerate(main_classes)}
idx2class = {i: k for i, k in enumerate(main_classes)}
print(class2idx)
print(idx2class)


# In[ ]:


# Define a consolidated dataset
probe_train_x = np.concatenate([np.array(dataset[k]) for k in main_classes], axis=0)
probe_train_y = np.concatenate([np.array([class2idx[k] for _ in range(len(dataset[k]))]) for k in main_classes])
print("Train set:", probe_train_x.shape, probe_train_y.shape)

# Fix the validation set to include the new attacks -- will collapse them to the same class right now
additional_val_classes = [f"backdoor_{attack_type}" for attack_type in attack_types]
main_classes_val = main_classes + additional_val_classes
print("Main validation classes:", main_classes_val)
class2idx_val = copy.deepcopy(class2idx)
class2idx.update({k: class2idx["backdoor"] for k in additional_val_classes})

starting_idx = np.max([v for k, v in class2idx.items()]) + 1
class2idx_val.update({k: starting_idx + idx for idx, k in enumerate(additional_val_classes)})
print("Class2idx updated:", class2idx)
print("Class2idx val:", class2idx_val)

probe_val_x = np.concatenate([np.array(dataset[f"{k}_val"]) for k in main_classes_val], axis=0)
probe_val_binary_y = np.concatenate([np.array([class2idx[k] for _ in range(len(dataset[f"{k}_val"]))]) for k in main_classes_val])
probe_val_y = np.concatenate([np.array([class2idx_val[k] for _ in range(len(dataset[f"{k}_val"]))]) for k in main_classes_val])
print("Validation set:", probe_val_x.shape, probe_val_binary_y.shape, probe_val_y.shape)


# In[ ]:

##
print("Training the trajectory classifier...")
n_neighbors = 20
clf = sklearn.neighbors.KNeighborsClassifier(n_neighbors)
clf.fit(probe_train_x, probe_train_y)


# In[ ]:


def plot_confusion_matrix_from_preds(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues, fontsize=15):
    cm = confusion_matrix(
        y_true,
        y_pred,
        sample_weight=None,
        labels=None,
        normalize=None,
    )
    
    if normalize:
        cm = cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
        cm = np.around(cm,decimals=2)
        cm[np.isnan(cm)] = 0.0
        print('Normalized confusion matrix')
    else:
        print('Confusion matrix, without normalization')

    plt.figure(figsize=(8, 7))
    
    im = plt.imshow(cm, interpolation='nearest', cmap=cmap)
    if title is not None:
        plt.title(title)
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=fontsize)
    
    tick_marks=np.arange(len(classes))
    display_labels = [x.title().replace("_", " ") for x in classes]
    plt.xticks(tick_marks, display_labels, fontsize=fontsize, rotation=45, ha="right")
    plt.yticks(tick_marks, display_labels, fontsize=fontsize, rotation=0, ha="right")
    
    thresh = cm.max() / 2
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", fontsize=fontsize, color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout()
        plt.ylabel('True label', fontsize=fontsize)
        plt.xlabel('Predicted label', fontsize=fontsize)


# In[ ]:


print("Evaluating the trajectory classifier...")
for include_all_val in [False, True]:
    if include_all_val:
        current_probe_val_x, current_probe_val_y, plot_classes = probe_val_x, probe_val_y, main_classes_val
    else:
        mask = probe_val_y < len(main_classes)
        print(f"Selecting {np.sum(mask)} probe examples for evaluating classifier without additional examples...")
        current_probe_val_x, current_probe_val_y, plot_classes = probe_val_x[mask], probe_val_y[mask], main_classes
    
    for normalize in [False, True]:
        prediction = clf.predict(current_probe_val_x)
        if include_all_val:  # Map predictions to all classes including additional ones
            additional_cls_mask = probe_val_y >= len(main_classes)
            correct_pred_mask = prediction == class2idx["backdoor"]
            full_mask = np.logical_and(additional_cls_mask, correct_pred_mask)
            prediction[full_mask] = current_probe_val_y[full_mask]  # Assign them to the actual corresponding class if they are correctly predicted as backdoors
        
        test_acc = (prediction == current_probe_val_y).astype(np.float32).mean()
        print(f"Evaluation results | Test: {100. * test_acc:.2f}%")
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        plot_confusion_matrix_from_preds(current_probe_val_y, prediction, plot_classes, normalize=normalize)
        plt.tight_layout()
        output_file = os.path.join(experiment_output_dir, f"probe_confusion_matrix_trajectories_val_probes{'_all' if include_all_val else ''}_{num_example_probes}{'_norm' if normalize else ''}.png")
        plt.savefig(output_file, dpi=300, bbox_inches="tight")


# In[ ]:


def assign_probe_classes_knn(clf, idx_train_loader, sorted_losses_all, idx2class, output_dir, class_to_surface, probe_class_to_surface):
    print("Computing probabilities for probe classes using kNN...")
    
    iterator = 0
    write_individual_img = True
    
    folder_counter = {}
    folder_queue = {}
    
    if output_dir is not None:
        for k in ref_probe_classes:
            output_loc = os.path.join(output_dir, k)
            if main_proc:
                print("Creating output location:", output_loc)
                if not os.path.exists(output_loc):
                    os.makedirs(output_loc)
            folder_counter[k] = 0
            folder_queue[k] = []
    
    for ((data, target), ex_idx) in idx_train_loader:
        for i in range(len(target)):
            if int(target[i]) != class_to_surface:
                continue
            print("Found instance of class:", class_to_surface)
            
            # Get example loss trajectory
            global_idx = int(ex_idx[i])
            print("Selected global idx:", global_idx)
            loss_traj = [sorted_losses_all[j][global_idx] for j in range(len(sorted_losses_all))]
            if loss_traj[0] is None or global_idx >= len(train_set):  # Probe example
                continue
            
            # Compute the probabilities of an example belonging to these different groups
            probs = clf.predict_proba(np.array([loss_traj]))  # Cast it into a batch
            pred = np.argmax(probs, axis=1)
            assert len(pred) == 1
            pred = pred[0]
            pred_prob = float(probs[0, pred])
            
            # Save the images to folder
            imgs = torch.nn.functional.interpolate(data.cpu(), size=(224, 224))
            imgs = np.transpose(imgs.numpy(), (0, 2, 3, 1))  # BCHW -> BHWC
            imgs = np.clip(imgs * 255, 0, 255).astype(np.uint8)

            cls_name = train_set.classes[int(target[i])]
            pred_folder = idx2class[pred]
            
            if pred_folder not in probe_class_to_surface:
                continue
            
            if output_dir is not None:
                if write_individual_img:
                    file_name = f"rank_{rank}_idx_{global_idx}_count_{folder_counter[pred_folder]}_conf_{pred_prob:.2f}_{cls_name}.png"
                    output_loc = os.path.join(output_dir, pred_folder, file_name)
                    img = imgs[i]
                    cv2.imwrite(output_loc, img[:, :, ::-1])  # RGB -> BGR

                    if iterator % 100 == 0:
                        print("Writing image to fle:", output_loc)

                    folder_counter[pred_folder] += 1
                else:
                    folder_queue[pred_folder].append((imgs[i], cls_name, probs[pred]))
                    if len(folder_queue[pred_folder]) == num_queue_plots:
                        file_name = f"rank_{rank}_idx_{global_idx}_count_{folder_counter[pred_folder]}_conf_{pred_prob:.2f}_{pred_folder}.png"
                        output_file = os.path.join(output_dir, pred_folder, file_name)
                        plot_probe_ex([x[0] for x in folder_queue[pred_folder]], [x[1] for x in folder_queue[pred_folder]],
                                    [x[2] for x in folder_queue[pred_folder]], output_file)
                        folder_counter[pred_folder] += 1
                        if iterator % 4 == 0:
                            print("Writing image to fle:", output_file)
                        iterator += 1
                        folder_queue[pred_folder] = []  # Empty the queue


# In[ ]:


surface_examples = False
if surface_examples:
    surface_dir = os.path.join(experiment_output_dir, f"surfaced_examples_{dataset}")
    if main_proc:
        if os.path.exists(surface_dir):
            shutil.rmtree(surface_dir)
        os.makedirs(surface_dir)
    dist_utils.wait_for_other_procs()
    
    train_set.transform = transforms.Compose(no_transform)
    if "cifar" in dataset or dataset == "mnist":
        classes_to_surface = list(range(num_classes))
    else:
        classes_to_surface = [531, 671, 728, 901, 999]  # Digital watch, mountain bike, plastic bag, Whiskey jug, Tiolet tissue,
        classes_to_surface += [407, 413, 417, 435, 465, 508, 510, 527]  # Ambulance, Assault gun, Baloon, Bath tub, Bulletproof vest, computer keyboard, container ship, desktop computer
        classes_to_surface += [982, 471, 651, 653, 771, 810, 859]  # Groom, canon, microwave, milk can, safe, space bar, toaster
        classes_to_surface += [954, 953, 919, 847, 657, 605, 569]  # Banana, pineapple, street sign, tank, missile, iPod, gas mask
    print("Chosen class:", classes_to_surface)

    for class_to_surface in classes_to_surface:
        print("Surfacing examples from class:", class_to_surface)
        output_path = os.path.join(surface_dir, f"complete_traj_train_cls_{class_to_surface}_{train_set.classes[class_to_surface]}")
        assign_probe_classes_knn(clf, new_idx_loader, sorted_losses_all, idx2class, output_path, class_to_surface, probe_class_to_surface=["backdoor"])
    print("All files saved. Execution completed!")

