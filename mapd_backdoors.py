#!/usr/bin/env python
# coding: utf-8

# ## Inserting probes into the model for inspecting model phase

# In[ ]:

##

import os
import sys
import math
import copy
import wandb
import pickle
import shutil
import random
import natsort
import warnings
from tqdm import tqdm
from collections import Counter

import numpy as np
import cv2
import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import sklearn.neighbors
import sklearn.cluster
import sklearn.metrics
from scipy.fftpack import dct


import dist_utils
from dataset_utils import get_settings_for_dataset, make_probe_dataset, make_index_dataset, \
    get_loader, IdxDataset
from plot_utils import plot_probe_examples, plot_probe_ex, some_plot, some_other_plot, make_normalizers, \
    yet_another_plot, one_more_plot, plot_loss_dynamics_and_violin, visualize_loss_trajectories, \
    visualize_loss_trajectories_specific, plot_confusion_matrix_from_preds, plot_attack_success_stats, \
    num_queue_plots, plot_auc, generate_embeddings_from_trajectories
from backdoors import make_train_probes, make_val_probes, make_test_probes
from torch_utils import get_model, get_optimizer, train, test, test_tensor, FreqCNN

default_attack  = "patch"
default_defense = "mapd"
default_poisoning_ratio = None


dataset_choices = ["mnist", "cifar10", "cifar100", "gtsrb", "imagenet"]
attack_choices  = ["all", "patch", "single_pix", "random", "fixed", "sinusoid", "warped", "narcissus"]  # TODO: Add sleeper...
defense_choices = ["mapd", "nc", "ac", "ss", "freq", "abl"]
poisoning_ratio_choices = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]

if len(sys.argv) < 2:
    print(f"Usage: {sys.argv[0]} <Dataset: {'/'.join(dataset_choices)}>")
    exit()

dataset = sys.argv[1]
assert dataset in dataset_choices

if len(sys.argv) >= 3:
    attack = sys.argv[2]
    assert attack in attack_choices
else:
    attack = default_attack


if len(sys.argv) >= 4:
    defense = sys.argv[3]
    assert defense in defense_choices
else:
    defense = default_defense

if len(sys.argv) >= 5:
    poisoning_ratio = float(sys.argv[4])
    assert poisoning_ratio in poisoning_ratio_choices
else:
    poisoning_ratio = default_poisoning_ratio

print(dataset, attack, defense, poisoning_ratio)

if attack == "all":
    train_probe_attack = "reversed_patch"
    val_probe_attacks = ["patch", "single_pix", "random", "fixed", "sinusoid", "warped"]
    # if dataset == "cifar10":  # TODO: Add sleeper fully...
    #     val_probe_attacks.append("sleeper")
elif attack in {"patch", "single_pix", "fixed", "sinusoid"}:
    train_probe_attack = "reversed_" + attack
    val_probe_attacks = [attack]
elif attack == "narcissus":
    train_probe_attack = "alt_narcissus"
    val_probe_attacks = [attack]
else:
    train_probe_attack = attack
    val_probe_attacks = [attack]
num_train_probes = 250  # Fixed number -- 4x this many probes will be made
                            # (now 3x this number of backdoor probes -- (num) normal, (num) mislabeled, (num) normal for val;
                            #  then (2*num) clean examples set aside for comparison.
train_probe_counts = [25, 50, 100, 150, 200, 250, 300, 400, 500]
num_train_probes = train_probe_counts[0]  # TODO: What if multiple of the same count are wanted?

if attack == "warped":
    num_train_probes = 1500  # More probes to more closely imitate learning dynamics of the larger warped attack.
if attack == "sinusoid" and dataset == "gtsrb":
    num_train_probes = 250  # Need to choose number of train probes carefully since the classes are so small -- this produces 500 poisoned.
# Fraction in terms of overall dataset size!! Not in terms of per-class size.
num_val_probes = {
    "patch": 0.01,
    "single_pix": 0.01,
    "random": 0.01,
    "fixed": 0.01,
    "sinusoid": 0.1,  # Clean label attacks are expressed as a fraction of the target class!
    "warped": 0.1,
    "sleeper": 0.05,  # TODO: Is this right? Checks out for CIFAR-10 I think, but it's 100%
    "narcissus": 0.005,  # So they claim... 25 images!!
}
if dataset == "gtsrb":
    num_val_probes["patch"] = 0.02
    num_val_probes["single_pix"] = 0.04
    num_val_probes["warped"] = 0.2
correct_abl = False  # If true, hard set poisoning ratio for abl to 10% at least.
if correct_abl and defense == "abl":
    poisoning_ratio = 0.1

if poisoning_ratio is not None:
    for k in num_val_probes:
        num_val_probes[k] = poisoning_ratio
num_test_probes = 10000


# Set random seed
seed = 3
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed=seed)
random.seed(seed)
torch.use_deterministic_algorithms(True)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

# Plotting config
include_plot_title = False
font_size = 16


# Essential config
log_predictions = True
distributed = True if dataset == "imagenet" else False
project_id = "exp45"
experiment_output_dir = f"./backdoor_{project_id}_{dataset}_{defense}_{attack}{'_' + str(poisoning_ratio) if poisoning_ratio is not None else ''}"
model_collection_dir = f"./backdoor_{project_id}_model_{dataset}_{attack}{'_' + defense if defense in {'mapd'} else ''}{'_' + str(poisoning_ratio) if poisoning_ratio is not None else ''}"
# model_collection_dir = experiment_output_dir
num_workers = 8
surface_examples = False

print("Dataset:", dataset)
print("Distributed training:", distributed)

# Initalize W&B -- assumes wandb is already logged in
log_wandb = False
if dist_utils.is_main_proc():
    print("Initializing w&b")
    wandb_project = f"mapd_backdoors_{dataset}"
    wandb_run_name = f"attack_{attack}_defense_{defense}{'_poisoning_ratio' + str(poisoning_ratio) if poisoning_ratio is not None else ''}_run_{project_id}"
    wandb.init(
        project=wandb_project,
        name=wandb_run_name,
    )
    log_wandb = True

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

dist_utils.setup_for_distributed(main_proc)
warnings.filterwarnings("ignore", "Warning: Leaking Caffe2 thread-pool after fork. (function pthreadpool)", UserWarning)


recompute_results = False
if main_proc:
    if not os.path.exists(model_collection_dir):
        os.makedirs(model_collection_dir)
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


# Standardizing nomenclature...
if defense == "mapd":
    attack_types = ["backdoor"] + [f"backdoor_{attack}" for attack in val_probe_attacks]
else:
    attack_types = [f"backdoor_{attack}" for attack in val_probe_attacks]

val_probes, attack_targets, random_pattern, warping_grids = make_val_probes(num_classes, dataset, train_set_wo_aug,
                                                                            num_val_probes, val_probe_attacks,
                                                                            experiment_output_dir, main_proc, img_size,
                                                                            device,)

include_val_probe_examples = False


train_probe, attack_target, aux_data = make_train_probes(num_classes, dataset, train_set_wo_aug,
                                                         num_train_probes, train_probe_attack,
                                                         experiment_output_dir, main_proc, img_size, device,
                                                         val_probe_indices=val_probes["all_backdoor_idx"],
                                                         include_val_probe_examples=include_val_probe_examples)

train_probes_idx = train_probe["all_backdoor_idx"]

test_probes = make_test_probes(test_set, dataset, num_test_probes, val_probe_attacks, attack_targets,
                               random_pattern, warping_grids, experiment_output_dir, main_proc, img_size, device)


# Merge probe dicts
if defense == "mapd":
    probes = {**train_probe, **val_probes}
    unified_backdoor_idx = np.concatenate((train_probe['all_backdoor_idx'], val_probes['all_backdoor_idx']))
    probes['all_backdoor_idx'] = unified_backdoor_idx
    chosen_attack_targets = {**{'backdoor': attack_target}, **attack_targets}
    plot_probe_examples(probes, dataset, train_set, attack_types, rank, experiment_output_dir, log_wandb=log_wandb)
else:
    probes = val_probes



# Hyperparameters
if dataset == "mnist":
    num_epochs = 25
    batch_size = 256
elif "cifar" in dataset:
    num_epochs = 100 if defense != "mapd" else 50
    batch_size = 128
else:
    assert dataset == "imagenet" or dataset == "gtsrb"
    num_epochs = 100 if defense != "mapd" else 50
    optimizer_batch_size = 256
    batch_size = 256
    if distributed:
        assert batch_size % world_size == 0
        batch_size = batch_size // world_size
        print(f"Optimizer batch size: {optimizer_batch_size} / World size: {world_size} / Local batch size: {batch_size}")
tensor_batch_size = batch_size if dataset == "gtsrb" else None
lr = 0.1
momentum = 0.9
wd = 0.0001
moving_avg_weight = None
augment_in_pretraining = False
augment_in_retraining = True
if augment_in_retraining == False:
    raise NotImplementedError("Set augment = False in dataset_utils instead, good luck.")


comb_train_set, comb_train_indices, dataset_probe_identity, discarded_idx = \
    make_probe_dataset(probes, train_set, dataset, num_train_probes, defense,
                       train_transform, val_probe_attacks, experiment_output_dir,
                       device, include_val_probe_examples=include_val_probe_examples)
valid_idx = [i for i in range(len(train_set)) if i not in discarded_idx]


model = get_model(dataset, num_classes, device, local_rank, verbose=False)
criterion, optimizer, lr_scheduler, scaler = get_optimizer(model, device, lr, momentum, wd, num_epochs)

new_idx_loader, new_idx_loader_wo_aug, test_idx_loader, idx_dataset = \
    make_index_dataset(comb_train_set, comb_train_indices, test_set,
                       no_transform, batch_size, distributed, num_workers)

# Load train probes onto gpu -- inexpensive and saves time.
# ...could load entire train set onto gpu (15GB tops in GTSRB), but that's a pain, code-wise.
to_cuda = copy.deepcopy(attack_types)
if defense == 'mapd':
    if include_val_probe_examples:
        to_cuda += ['backdoor_val', 'clean', 'clean_val']
    else:
        to_cuda += ['clean']

for key in to_cuda:
    probes[key] = probes[key].to(device)
    probes[f'{key}_labels'] = probes[f'{key}_labels'].to(device)
# Non-GTSRB test probes can stay on the GPU
if dataset != "gtsrb":
    for key in test_probes:
        test_probes[key] = test_probes[key].to(device)


model_dir = os.path.join(model_collection_dir, f"models_{dataset}")
model_file = os.path.join(model_dir, f"model_{dataset}_dynamics.pth")
data_file = os.path.join(model_collection_dir, f"stats_{dataset}_dynamics.pkl")
data_statistics_file = os.path.join(model_collection_dir, f"stats_{dataset}_data_statistics.pkl")


# In[ ]:


if main_proc and not os.path.exists(model_dir):
    os.mkdir(model_dir)


# In[ ]:


ref_probe_classes = ["backdoor", "clean"]
label_map_dict = {"backdoor": "Backdoor (probe)",
                  "backdoor_val": "Backdoor (probe) [Val]",
                  "clean": "Clean",
                  "clean_val": "Clean [Val]",
                  "backdoor_patch_val": "Backdoor (Patch)",
                  "backdoor_single_pix_val": "Backdoor (Single pixel patch)",
                  "backdoor_reversed_val": "Backdoor (Reversed)",
                  "backdoor_reversed_single_pix_val": "Backdoor (Reversed single pixel)",
                  "backdoor_random_val": "Backdoor (Blend-R)",
                  "backdoor_fixed_val": "Backdoor (Blend-P)",
                  "backdoor_sinusoid_val": "Backdoor (Sinusoid)",
                  "backdoor_random_boosted_val": "Backdoor (Blend-R; boosted)",
                  "backdoor_fixed_boosted_val": "Backdoor (Blend-P; boosted)",
                  "backdoor_sinusoid_boosted_val": "Backdoor (Sinusoid; boosted)",
                  "backdoor_warped_val": "Backdoor (Warped)",
                  "train": "Train",
                  "test": "Test"}


# In[ ]:

def log_results_and_update_stats_and_preds(log_predictions, model, device, criterion, test_idx_loader, distributed,
                                           rank, new_idx_loader_wo_aug, attack_types, probes, val_probes, defense,
                                           tensor_batch_size, epoch, statistics=None, predictions=None, use_eval_mode=True,
                                           max_loss_val_bound=None):
    test_stats, test_preds = test(model, device, criterion, test_idx_loader, distributed, rank,
                                  log_predictions=log_predictions)
    if statistics is not None:
        statistics["test"].append(test_stats)
        if log_wandb:
            wandb.log({wandb_prefix+"test": test_stats})

    if log_predictions:
        # Don't use train_idx_loader here -- also assumes that probes are include for later evaluation
        train_stats, train_preds = test(model, device, criterion, new_idx_loader_wo_aug, distributed, rank,
                                        set_name="Train", log_predictions=log_predictions, use_eval_mode=use_eval_mode,
                                        max_loss_val_bound=max_loss_val_bound)
        if statistics is not None:
            statistics["train"].append(train_stats)
            if log_wandb:
                wandb.log({wandb_prefix+"train": train_stats})

        # Add predictions from all the different sets / probes
        if predictions is not None:
            predictions[epoch] = {}  # Dict of dict
            predictions[epoch]["train"] = train_preds
            predictions[epoch]["test"] = test_preds

    # Collect probe statistics
    if defense == "mapd":
        atks = attack_types + ["clean"]
    else:
        atks = attack_types

    for attack_type in atks:
        if attack_type in {'clean', 'backdoor'}:
            stats, preds = test_tensor(model, device, criterion, probes[attack_type],
                                       probes[f"{attack_type}_labels"],
                                       msg=f"{attack_type.capitalize().replace('_', ' ')} probe",
                                       log_predictions=log_predictions, batch_size=tensor_batch_size,
                                       use_eval_mode=use_eval_mode, max_loss_val_bound=max_loss_val_bound)
            val_stats, val_preds = None, None
            if attack_type+"_val" in probes:
                val_stats, val_preds = test_tensor(model, device, criterion, probes[attack_type+"_val"],
                                                probes[f"{attack_type}_val_labels"],
                                                msg=f"{attack_type.capitalize().replace('_', ' ')} probe (val)",
                                                log_predictions=log_predictions, batch_size=tensor_batch_size,
                                                use_eval_mode=use_eval_mode, max_loss_val_bound=max_loss_val_bound)
            if predictions is not None:
                predictions[epoch][attack_type] = preds
                if val_preds is not None:
                    predictions[epoch][attack_type+"_val"] = val_preds
            if statistics is not None:
                statistics[attack_type].append(stats)
                if log_wandb:
                    wandb.log({wandb_prefix+attack_type: stats})
                if val_stats is not None:
                    statistics[attack_type+"_val"].append(val_stats)
                    if log_wandb:
                        wandb.log({wandb_prefix+attack_type+"_val": val_stats})
        else:
            suffix = ' (val)'
            stats, preds = test_tensor(model, device, criterion, probes[attack_type],
                                       probes[f"{attack_type}_labels"],
                                       msg=f"{attack_type.capitalize().replace('_', ' ')} probe{suffix}",
                                       log_predictions=log_predictions, batch_size=tensor_batch_size,
                                       use_eval_mode=use_eval_mode, max_loss_val_bound=max_loss_val_bound)
            if predictions is not None:
                predictions[epoch][attack_type] = preds
            if statistics is not None:
                statistics[attack_type].append(stats)
            if log_wandb:
                wandb.log({wandb_prefix+attack_type: stats})


def test_unseen_probes(log_predictions, model, device, criterion, test_probes, val_probe_attacks, tensor_batch_size):
    output_dict = {}
    attacks = [x for x in test_probes.keys() if not x.endswith("_labels")]
    for attack in attacks:
        stats, _ = test_tensor(model, device, criterion, test_probes[attack],
                               test_probes[f"{attack}_labels"],
                               msg=f"{attack.capitalize().replace('_', ' ')} probe (test; unseen)",
                               log_predictions=log_predictions, batch_size=tensor_batch_size)
        output_dict[attack] = {}
        output_dict[attack]['accuracy'] = stats['acc']
        output_dict[attack]['total'] = stats['total']
        output_dict[attack]['correct'] = stats['correct']
    if log_wandb:
        wandb.log({"unseen_probes": output_dict})
    return output_dict


wandb_prefix = ''
if defense in {"nc", "ac", "ss", "freq", "abl"}:
    if not os.path.exists(model_file):
        loader = new_idx_loader if augment_in_pretraining else new_idx_loader_wo_aug
        for epoch in range(num_epochs):
            train(model, device, loader, optimizer, criterion, scaler)
            if (epoch + 1) % 5 == 0:
                print(f"Stats for epoch {epoch + 1}")
                log_results_and_update_stats_and_preds(log_predictions, model, device, criterion, test_idx_loader,
                                                       distributed, rank, new_idx_loader_wo_aug, attack_types, probes,
                                                       val_probes, defense, tensor_batch_size, epoch)
                test_unseen_probes(log_predictions, model, device, criterion, test_probes, val_probe_attacks,
                                   tensor_batch_size)
                if main_proc:
                    # Save the model
                    model_file_base, model_file_ext = os.path.splitext(model_file)
                    current_model_file = f"{model_file_base}_ep_{epoch}{model_file_ext}"
                    # torch.save(model.state_dict(), current_model_file)

            if lr_scheduler is not None:
                lr_scheduler.step()

        if main_proc:
            # Save the model
            torch.save(model.state_dict(), model_file)
    else:
        assert os.path.exists(model_file)
        print("Data files already found. Loading data from saved checkpoints...")

        model.load_state_dict(torch.load(model_file, map_location=device))
elif defense == "mapd":
    stats_dict = {n: {} for n in train_probe_counts}
    for num_train_probes in train_probe_counts:
        wandb_prefix = f"num_train_probes_{num_train_probes}_"
        if num_train_probes != train_probe_counts[0]:
            val_probes, attack_targets, random_pattern, warping_grids = make_val_probes(num_classes, dataset,
                                                                                        train_set_wo_aug,
                                                                                        num_val_probes,
                                                                                        val_probe_attacks,
                                                                                        experiment_output_dir,
                                                                                        main_proc, img_size,
                                                                                        device, )

            include_val_probe_examples = False

            train_probe, attack_target, aux_data = make_train_probes(num_classes, dataset, train_set_wo_aug,
                                                                     num_train_probes, train_probe_attack,
                                                                     experiment_output_dir, main_proc, img_size, device,
                                                                     val_probe_indices=val_probes["all_backdoor_idx"],
                                                                     include_val_probe_examples=include_val_probe_examples)

            train_probes_idx = train_probe["all_backdoor_idx"]

            test_probes = make_test_probes(test_set, dataset, num_test_probes, val_probe_attacks, attack_targets,
                                           random_pattern, warping_grids, experiment_output_dir, main_proc, img_size,
                                           device)

            probes = {**train_probe, **val_probes}
            unified_backdoor_idx = np.concatenate((train_probe['all_backdoor_idx'], val_probes['all_backdoor_idx']))
            probes['all_backdoor_idx'] = unified_backdoor_idx
            chosen_attack_targets = {**{'backdoor': attack_target}, **attack_targets}

            comb_train_set, comb_train_indices, dataset_probe_identity, discarded_idx = \
                make_probe_dataset(probes, train_set, dataset, num_train_probes, defense,
                                   train_transform, val_probe_attacks, experiment_output_dir,
                                   device, include_val_probe_examples=include_val_probe_examples)
            valid_idx = [i for i in range(len(train_set)) if i not in discarded_idx]

            model = get_model(dataset, num_classes, device, local_rank, verbose=False)
            criterion, optimizer, lr_scheduler, scaler = get_optimizer(model, device, lr, momentum, wd, num_epochs)

            new_idx_loader, new_idx_loader_wo_aug, test_idx_loader, idx_dataset = \
                make_index_dataset(comb_train_set, comb_train_indices, test_set,
                                   no_transform, batch_size, distributed, num_workers)

            # Load train probes onto gpu -- inexpensive and saves time.
            # ...could load entire train set onto gpu (15GB tops in GTSRB), but that's a pain, code-wise.
            to_cuda = copy.deepcopy(attack_types)
            if defense == 'mapd':
                if include_val_probe_examples:
                    to_cuda += ['backdoor_val', 'clean', 'clean_val']
                else:
                    to_cuda += ['clean']

            for key in to_cuda:
                probes[key] = probes[key].to(device)
                probes[f'{key}_labels'] = probes[f'{key}_labels'].to(device)
            # Non-GTSRB test probes can stay on the GPU
            if dataset != "gtsrb":
                for key in test_probes:
                    test_probes[key] = test_probes[key].to(device)

        model_file = os.path.join(model_dir, f"model_{dataset}_{num_train_probes}_probes_dynamics.pth")
        data_file = os.path.join(model_collection_dir, f"stats_{dataset}_{num_train_probes}_probes_dynamics.pkl")

        if os.path.exists(model_file):
            assert os.path.exists(data_file)
            print(f"Data files for {num_train_probes} probes already found. Loading data from saved checkpoints...")
            model.load_state_dict(torch.load(model_file, map_location=device))
            assert os.path.exists(data_file)
            with open(data_file, "rb") as f:
                stats_dict[num_train_probes] = pickle.load(f)

        else:
            statistics = {"train": [], "test": []}
            statistics.update({k: [] for k in attack_types + ['clean']})
            if include_val_probe_examples:
                statistics.update({k+"_val": [] for k in ref_probe_classes})
            inv_probe_map = {i: v for i, v in enumerate(ref_probe_classes)}

            predictions = {}
            save_models = False
            uniform_dist_perplex = -math.log(1/num_classes)
            max_loss_val_bound = None # 2 * uniform_dist_perplex  # equal to twice the entropy of a uniform distribution over classes
            use_eval_mode = True  # eval mode BN
            print(f"!! Using max loss bound: {max_loss_val_bound} / Eval mode: {use_eval_mode}")
            moving_avg_weight = None
            loader = new_idx_loader if augment_in_pretraining else new_idx_loader_wo_aug

            for epoch in range(num_epochs):
                output_dict = train(model, device, loader, optimizer, criterion, scaler)

                # Collect test set statistics
                print("Stats for epoch #", epoch+1)
                log_results_and_update_stats_and_preds(log_predictions, model, device, criterion, test_idx_loader, distributed,
                                                       rank, new_idx_loader_wo_aug, attack_types, probes, val_probes, defense,
                                                       tensor_batch_size, epoch, statistics=statistics, predictions=predictions,
                                                       use_eval_mode=use_eval_mode, max_loss_val_bound=max_loss_val_bound)

                if epoch % 5 == 4:
                    test_unseen_probes(log_predictions, model, device, criterion, test_probes, val_probe_attacks,
                                       tensor_batch_size)
                if lr_scheduler is not None:
                    lr_scheduler.step()

                if main_proc and save_models:
                    # Save the model
                    model_file_base, model_file_ext = os.path.splitext(model_file)
                    current_model_file = f"{model_file_base}_ep_{epoch}{model_file_ext}"
                    torch.save(model.state_dict(), current_model_file)

                # Close all figures
                plt.close('all')

            test_unseen_probes(log_predictions, model, device, criterion, test_probes, val_probe_attacks, tensor_batch_size)

            if log_predictions:
                statistics["predictions"] = predictions

            statistics['aux'] = {'dataset_probe_identity': dataset_probe_identity,
                                 'discarded_idx': discarded_idx,
                                 'valid_idx': valid_idx}

            if main_proc:
                # Save the model
                torch.save(model.state_dict(), model_file)

                # Save the final data
                with open(data_file, "wb") as f:
                    pickle.dump(statistics, f, protocol=pickle.HIGHEST_PROTOCOL)
            stats_dict[num_train_probes] = statistics
else:
    raise NotImplementedError

print("Final model performance:")
test(model, device, criterion, test_idx_loader, distributed, rank, log_predictions=log_predictions)
test_unseen_probes(log_predictions, model, device, criterion, test_probes, val_probe_attacks, tensor_batch_size)




if defense == "mapd":
    knn_classifiers = {}
    oc_knn_classifiers = {}
    for key in stats_dict:
        num_train_probes = key
        statistics = stats_dict[key]
        dataset_probe_identity, discarded_idx, valid_idx = statistics['aux']['dataset_probe_identity'], statistics['aux']['discarded_idx'], statistics['aux']['valid_idx']

        print(f"===== {num_train_probes} probes =====")
        data_statistics_file = os.path.join(model_collection_dir, f"stats_{dataset}_{num_train_probes}_probes_data_statistics.pkl")
        model_file = os.path.join(model_dir, f"model_{dataset}_{num_train_probes}_probes_dynamics.pth")
        model.load_state_dict(torch.load(model_file, map_location=device))

        print("Final train accuracy:", statistics["train"][-1])
        print("Final test accuracy:", statistics["test"][-1])
        print("Keys in statistics file:", natsort.natsorted(list(statistics.keys())))

        # Just pretend they're all named _val for convenience sakes...
        for key in list(statistics.keys()):
            if "backdoor" in key and key != "backdoor" and "_val" not in key:
                new_key = key + "_val"
                statistics[new_key] = statistics[key]
                del statistics[key]

        print("Keys in statistics file:", natsort.natsorted(list(statistics.keys())))

        some_plot(statistics, log_predictions, label_map_dict, include_plot_title, dataset, main_proc, experiment_output_dir, log_wandb=log_wandb)
        some_other_plot(statistics, log_predictions, label_map_dict, include_plot_title, dataset, main_proc, experiment_output_dir, num_train_probes, log_wandb=log_wandb)


        if not log_predictions:
            print("Can't compute other statistics without the model predictions...")
            exit()


        # ### Learning dynamics per example


        unique_probe_identity = np.unique(dataset_probe_identity)
        print("Unique dataset probe identity:", unique_probe_identity)


        if not os.path.exists(data_statistics_file):
            sorted_ex_list = []
            num_total_vals = len(dataset_probe_identity)  # 50000 + 600

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
                         label_map_dict, include_plot_title, dataset, main_proc, experiment_output_dir, log_wandb=log_wandb)

        # ### Loss distribution plots

        # List of example_idx at different epochs i.e. [epoch_1_loss_vals, ...., epoch_n_loss_vals]
        ex_idx = [statistics["predictions"][i]["train"]["ex_idx"] for i in range(len(statistics["predictions"]))]
        loss_values = [statistics["predictions"][i]["train"]["loss"] for i in range(len(statistics["predictions"]))]
        print(len(ex_idx), len(loss_values))


        sorted_losses_all = []

        print("Computing the sorted loss list...")
        normalize_trajectory = False
        for i in range(len(ex_idx)):  # Iterate over the epochs
            current_ex_idx = ex_idx[i]
            current_loss_vals = loss_values[i]
            assert len(current_ex_idx) == len(current_loss_vals), f"{len(current_ex_idx)} != {len(current_loss_vals)}"
            current_sorted_loss_vals = [None for _ in range(len(dataset_probe_identity))]  # Includes both the training set as well as the probes i.e. len(comb_train_set)
            for j, k in enumerate(current_ex_idx):
                current_sorted_loss_vals[k] = current_loss_vals[j]

            if normalize_trajectory:
                new_vals = np.array(current_sorted_loss_vals)
                new_vals[new_vals != None] = (new_vals[new_vals != None] - new_vals[new_vals != None].mean()) / new_vals[new_vals != None].std()
                current_sorted_loss_vals = new_vals.tolist()
            elif moving_avg_weight is not None and i > 0:
                old_vals = np.array(sorted_losses_all[-1])
                new_vals = np.array(current_sorted_loss_vals)
                new_vals[new_vals != None] = moving_avg_weight * old_vals[old_vals != None] + (1. - moving_avg_weight) * new_vals[new_vals != None]
                current_sorted_loss_vals = new_vals.tolist()
            sorted_losses_all.append(current_sorted_loss_vals)


        class_names = list(np.unique(dataset_probe_identity))
        print(class_names)


        one_more_plot(sorted_losses_all, class_names, label_map_dict, dataset_probe_identity,
                          dataset, main_proc, experiment_output_dir, log_wandb=log_wandb)


        plot_loss_dynamics_and_violin(sorted_losses_all, class_names, label_map_dict, dataset_probe_identity,
                                          dataset, experiment_output_dir, main_proc, log_wandb=log_wandb)


        if poisoning_ratio is None:
            for val_included in [True, False]:
                visualize_loss_trajectories(class_names, label_map_dict, dataset_probe_identity,
                                            sorted_losses_all, experiment_output_dir, main_proc, dataset,
                                            val_included=val_included, clf=None, output_file=None)
            visualize_loss_trajectories_specific(class_names, label_map_dict, dataset_probe_identity,
                                                 sorted_losses_all, experiment_output_dir, main_proc,
                                                 dataset, output_file=None)
            generate_tsne_plot = False
            if generate_tsne_plot:
                generate_embeddings_from_trajectories(class_names, label_map_dict, dataset_probe_identity,
                                                    sorted_losses_all, experiment_output_dir, main_proc,
                                                    dataset, output_file=None, embedding_type='tsne', log_wandb=log_wandb)

        # Convert the data into a complete trajectory dataset
        print("Converting trajectories to dataset...")
        # Note, this was initially named 'dataset', which had some name conflicts...
        traj_dataset = {}
        for i, cls in enumerate(class_names):
            relevant_idx = [i for i in range(len(dataset_probe_identity)) if dataset_probe_identity[i] == cls]
            print(f"Class: {cls} / # relevant idx: {len(relevant_idx)}")

            traj_dataset[cls] = []
            empty_idx = []
            for j in range(len(relevant_idx)):
                if sorted_losses_all[0][relevant_idx[j]] is None:  # The whole trajectory should be none since these examples are used in probes
                    assert all([sorted_losses_all[epoch][relevant_idx[j]] is None for epoch in range(len(sorted_losses_all))])
                    empty_idx.append(relevant_idx[j])
                    continue
                trajectory = [float(sorted_losses_all[epoch][relevant_idx[j]]) for epoch in range(len(sorted_losses_all))]
                traj_dataset[cls].append(trajectory)
            assert len(traj_dataset[cls]) == len(relevant_idx) - len(empty_idx)
            if len(empty_idx) > 0:
                print("Number of empty trajectories:", len(empty_idx))

        print("Total number of keys found:", traj_dataset.keys(), {k: len(traj_dataset[k]) for k in traj_dataset.keys()})
        trajectory_dataset_file = os.path.join(experiment_output_dir, f"loss_trajectories.pkl")
        with open(trajectory_dataset_file, "wb") as f:
            pickle.dump(traj_dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("Trajectory dataset written to file:", trajectory_dataset_file)


        # In[ ]:


        print("Converting trajectories to numpy dataset...")
        class_names = natsort.natsorted(list(traj_dataset.keys()))
        print("Class names:", class_names)

        main_classes = ['backdoor', 'clean']
        print(main_classes)

        class2idx = {k: i for i, k in enumerate(main_classes)}
        idx2class = {i: k for i, k in enumerate(main_classes)}
        print(class2idx)
        print(idx2class)


        # In[ ]:


        # Define a consolidated dataset
        probe_train_x = np.concatenate([np.array(traj_dataset[k]) for k in main_classes], axis=0)
        probe_train_y = np.concatenate([np.array([class2idx[k] for _ in range(len(traj_dataset[k]))]) for k in main_classes])
        print("Train set:", probe_train_x.shape, probe_train_y.shape)

        # Fix the validation set to include the new attacks -- will collapse them to the same class right now
        additional_val_classes = [attack for attack in attack_types if attack != "backdoor"]  # New backdoor style
        main_classes_val = main_classes + additional_val_classes
        print("Main validation classes:", main_classes_val)
        class2idx_val = copy.deepcopy(class2idx)
        class2idx.update({k: class2idx["backdoor"] for k in additional_val_classes})

        starting_idx = np.max([v for k, v in class2idx.items()]) + 1
        class2idx_val.update({k: starting_idx + idx for idx, k in enumerate(additional_val_classes)})
        print("Class2idx updated:", class2idx)
        print("Class2idx val:", class2idx_val)

        probe_val_x = np.concatenate([np.array(traj_dataset[f"{k}_val"] if f"{k}_val" in traj_dataset else traj_dataset[k]) for k in main_classes_val], axis=0)
        probe_val_binary_y = np.concatenate([np.array([class2idx[k] for _ in range(len(traj_dataset[f"{k}_val"] if f"{k}_val" in traj_dataset else traj_dataset[k]))]) for k in main_classes_val])
        probe_val_y = np.concatenate([np.array([class2idx_val[k] for _ in range(len(traj_dataset[f"{k}_val"] if f"{k}_val" in traj_dataset else traj_dataset[k]))]) for k in main_classes_val])
        print("Validation set:", probe_val_x.shape, probe_val_binary_y.shape, probe_val_y.shape)


        print("Training the trajectory classifier...")
        n_neighbors = 20  # TODO: Change the number of nearest neighbors here
        clf = sklearn.neighbors.KNeighborsClassifier(n_neighbors)
        clf.fit(probe_train_x, probe_train_y)
        knn_classifiers[num_train_probes] = clf

        # Create the one-class classifier
        clean_trajectories = traj_dataset["clean"]
        clean_labels = np.array([0 for _ in range(len(clean_trajectories))])
        oc_clf_neighbors = len(clean_trajectories)
        oc_clf = sklearn.neighbors.KNeighborsClassifier(oc_clf_neighbors)
        oc_clf.fit(clean_trajectories, clean_labels)
        oc_knn_classifiers[num_train_probes] = clf


        print("Evaluating the trajectory classifier...")
        # TODO!!! Because there's no explicit clean_val or backdoor_val, there's nothing here that's assigned to
        #   clean_val or backdoor_val, so the mask ends up all False. Currently breaks here!
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
                plot_confusion_matrix_from_preds(current_probe_val_y, prediction, plot_classes, include_all_val,
                                                 num_train_probes, experiment_output_dir, normalize=normalize, log_wandb=log_wandb)


        # In[ ]:

        # Plot detection TPR vs. FPR
        print("Generating AUC plot...")
        traj_preds = clf.predict_proba(probe_val_x)
        assert len(traj_preds.shape) == 2, traj_preds.shape

        neighbors_dist, _ = oc_clf.kneighbors(probe_val_x)
        assert neighbors_dist.shape == (len(probe_val_x), oc_clf_neighbors), neighbors_dist.shape
        avg_dist = neighbors_dist.mean(axis=1)
        avg_dist = -avg_dist  # Transform the distance as clean is class 1 and pred should represent the score for class 1

        # Plot the RoC curve
        key_list = ["MAP-D", "MAP-D (only clean)"]
        probe_val_y_mapped = probe_val_y.copy()
        clean_idx = class2idx["clean"]
        backdoor_idx = class2idx["backdoor"]
        print(f"!! Clean idx: {clean_idx} / Backdoor idx: {backdoor_idx}")

        probe_val_y_mapped[probe_val_y_mapped != clean_idx] = backdoor_idx
        print("Difference between probe val y and mapped probe val y", np.sum(probe_val_y != probe_val_y_mapped))

        label_dict = {"MAP-D": probe_val_y_mapped}  # Already contains both 0s and 1s appropriate for this task
        pred_dict = {"MAP-D": traj_preds[:, 1]}  # Probability of the label being 1 -- targets are also 1

        label_dict["MAP-D (only clean)"] = label_dict["MAP-D"].copy()
        pred_dict["MAP-D (only clean)"] = avg_dist

        output_file = os.path.join(experiment_output_dir, f"auc_{dataset}_clean_vs_backdoor.png")
        print("Base AUC")
        plot_auc(label_dict, pred_dict, key_list, output_file, log_wandb=log_wandb)

        for current_cls in main_classes_val:
            if current_cls == "clean":
                continue

            cls_idx = class2idx_val[current_cls]
            mask = np.logical_or(probe_val_y == cls_idx, probe_val_y == clean_idx)

            selected_probe_val_y = probe_val_y[mask]
            selected_probe_val_y[selected_probe_val_y == cls_idx] = backdoor_idx
            selected_traj_preds = traj_preds[mask]
            selected_oc_dist = avg_dist[mask]

            label_dict = {"MAP-D": selected_probe_val_y}  # Already contains both 0s and 1s appropriate for this task
            pred_dict = {"MAP-D": selected_traj_preds[:, 1]}  # Probability of the label being 1 -- targets are also 1

            label_dict["MAP-D (only clean)"] = label_dict["MAP-D"].copy()
            pred_dict["MAP-D (only clean)"] = selected_oc_dist

            output_file = os.path.join(experiment_output_dir, f"auc_{dataset}_clean_vs_backdoor_{current_cls}.png")
            print(f"{current_cls} AUC")
            plot_auc(label_dict, pred_dict, key_list, output_file, log_wandb=log_wandb)

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
                                            [x[2] for x in folder_queue[pred_folder]], output_file, log_wandb=log_wandb)
                                folder_counter[pred_folder] += 1
                                if iterator % 4 == 0:
                                    print("Writing image to fle:", output_file)
                                iterator += 1
                                folder_queue[pred_folder] = []  # Empty the queue


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


        def train_model(clean_model, loader, optimizer, criterion, scaler, lr_scheduler, output_checkpoint):
            for _ in range(num_epochs):
                train(clean_model, device, loader, optimizer, criterion, scaler)
                if lr_scheduler is not None:
                    lr_scheduler.step()
            torch.save(clean_model.state_dict(), output_checkpoint)
            print("!! Final checkpoint written to file:", output_checkpoint)


        def add_clean_to_output_dict(output_dict, model, device, criterion, test_loader, distributed, rank,
                                     log_predictions):
            out, _ = test(model, device, criterion, test_loader, distributed, rank, log_predictions=log_predictions)
            output_dict['clean'] = {}
            output_dict['clean']['accuracy'] = out['acc']
            output_dict['clean']['total'] = out['total']
            output_dict['clean']['correct'] = out['correct']

        # In[ ]:
        # Evaluate the attack success rate
        output_dict = test_unseen_probes(log_predictions, model, device, criterion, test_probes, val_probe_attacks,
                                         tensor_batch_size)
        add_clean_to_output_dict(output_dict, model, device, criterion, test_idx_loader, distributed, rank,
                                 log_predictions)
        output_file = os.path.join(experiment_output_dir, f"attack_success_initial.png")
        print(output_dict)
        plot_attack_success_stats(output_dict, label_map_dict, ref_probe_classes, output_file, title="Initial model", log_wandb=log_wandb)

    # In[ ]:
    print(knn_classifiers)
    print(oc_knn_classifiers)

    print("!! Collecting clean training indices...")
    losses_np = np.array(sorted_losses_all).transpose().astype(np.float64)  # Should be in format (# ex \times # epochs)
    missing_vals = np.isnan(losses_np).any(axis=1)  # Identify probe examples
    missing_vals_idx = np.where(missing_vals)[0]
    available_ex = np.logical_not(missing_vals)
    num_missing_vals = np.sum(missing_vals)
    print(f"!! Total loss traj len: {len(losses_np)} / Missing vals in loss trajs: {num_missing_vals}")
    take_weighted_average = True
    use_one_class = True
    use_exp_weighting = True
    oc_eps = 0.01
    all_probs = []
    all_dists = []

    if use_one_class:
        take_weighted_average = False
        for c in oc_knn_classifiers:
            clf = oc_knn_classifiers[c]
            dists = clf.kneighbors(losses_np[available_ex])[0]
            if use_exp_weighting:
                exp_weighted_dists = np.log(oc_eps + dists.mean(axis=1))
                zero_min_dists = exp_weighted_dists - exp_weighted_dists.min()
                probs = zero_min_dists / zero_min_dists.max()
            else:
                probs = (dists - dists.min()) / (dists - dists.min()).max()  # Min 0, max 1 -- extremely naive.
            all_probs.append(probs)
        all_probs = np.array(all_probs).reshape((*np.array(all_probs).shape, 1))

    else:
        for c in knn_classifiers:
            clf = knn_classifiers[c]
            all_probs.append(clf.predict_proba(losses_np[available_ex]))
            if take_weighted_average:
                all_dists.append(clf.kneighbors(losses_np[available_ex], clf.n_samples_fit_)[0].mean(axis=1))

    if take_weighted_average:
        avail_ex_probs = (np.array(all_probs)[:, :, 0] * (np.array(all_dists) / np.array(all_dists).sum(axis=0))).sum(axis=0)  # TODO: Weight the distances somehow...
    else:
        avail_ex_probs = np.array(all_probs)[:, :, 0].mean(axis=0)

    avail_ex_probs = np.array([avail_ex_probs, 1-avail_ex_probs]).T

    all_ex_probs = np.zeros((len(losses_np), 2), dtype=avail_ex_probs.dtype)
    all_ex_probs[available_ex] = avail_ex_probs
    all_ex_probs[missing_vals] = 1.1  # Always marked as probes and removed

    evaluate_classifier_on_training_probes = False
    if not evaluate_classifier_on_training_probes:
        print("!! Including training probe examples with their clean labels for retraining...")
        all_probe_original_idx = train_probes_idx
        train_probe_names = ["backdoor", "clean", "backdoor_val", "clean_val"]

        selected_idx = [(i, x) for i, x in enumerate(dataset_probe_identity) if x in train_probe_names]
        assert len(selected_idx) == len(all_probe_original_idx), f"{len(selected_idx)} == {len(all_probe_original_idx)}"
        all_probe_new_idx = [x[0] for x in selected_idx]

        # Include all probe idx -- remove their corrupted conunterparts as part of probe examples
        all_ex_probs[all_probe_original_idx] = 0.  # Always marked as clean and included
        all_ex_probs[all_probe_new_idx] = 1.1  # Always marked as probes and removed

    assert all_ex_probs.shape == (len(losses_np), 2), all_ex_probs.shape
    print("Output probs shape:", all_ex_probs.shape)
    # In[ ]:
    thresh_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]  # [0.25] if dataset == "imagenet" else [0.1, 0.25, 0.5, 0.75, 0.9]
    print("Threshold list:", thresh_list)
    num_epochs = 100

    output_checkpoint_dir = os.path.join(experiment_output_dir, "model_ft")
    if not os.path.exists(output_checkpoint_dir):
        os.makedirs(output_checkpoint_dir)
        print("!! Checkpoint output directory created:", output_checkpoint_dir)

    # train_types = ["original", "cleaned", "random"]
    train_types = ["cleaned"]

    for train_type in train_types:
        print("=" * 100)
        print(f"!! Using {train_type} training set....")

        current_thresh_list = [None] if train_type == "original" else thresh_list
        for threshold in current_thresh_list:
            if train_type == "original":
                # Use the training set w/o attacks
                assert threshold is None, threshold
                new_train_set_dl = get_loader(IdxDataset(train_set_wo_aug), distributed=distributed,
                                              num_workers=num_workers, batch_size=batch_size)
                title = "Retraining on the original train set (w/o backdoors)"
            else:
                assert threshold is not None, threshold
                is_clean = all_ex_probs[:, backdoor_idx] <= threshold  # probability of an example being the backdoor is less than thresh
                clean_indices = np.where(is_clean)[0]

                if train_type == "cleaned":  # Remove examples marked as backdoors
                    print(f"!! [Dataset cleansing] Total examples: {len(is_clean)} / # clean indices: {len(clean_indices)}")
                    discarded_indices = [i for i in range(len(all_ex_probs)) if i not in clean_indices and
                                         i not in missing_vals_idx]
                    probe_identity_discarded_samples = [dataset_probe_identity[i] for i in discarded_indices]
                    print("!! Discarded example identities:", Counter(probe_identity_discarded_samples))
                    
                    new_train_set_dl = get_loader(idx_dataset, indices=clean_indices, distributed=distributed,
                                                  num_workers=num_workers, batch_size=batch_size)
                    selected_indices = clean_indices
                else:
                    assert train_type == "random", train_type
                    selected_indices = np.random.choice(np.arange(len(idx_dataset)), size=(len(clean_indices),),
                                                        replace=False)

                num_avail_ex = len(is_clean) - num_missing_vals
                title = f"{'Random' if train_type == 'random' else 'Clean'} idx retraining (thresh={threshold:.1f}) [Total={len(is_clean)} / Selected: {len(selected_indices)}]"
                print(
                    f"!! Train type: {train_type} / Threshold: {threshold} / Total examples: {num_avail_ex} / # selected indices: {len(selected_indices)}")
                new_train_set_dl = get_loader(idx_dataset, indices=selected_indices, distributed=distributed,
                                              num_workers=num_workers, batch_size=batch_size)

            clean_model = get_model(dataset, num_classes, device, local_rank, verbose=False)
            criterion, optimizer, lr_scheduler, scaler = get_optimizer(clean_model, device, lr, momentum, wd,
                                                                       num_epochs)
            postfix = ""
            if threshold is not None:
                postfix = f"_thresh_{threshold:.2f}"
            output_checkpoint = os.path.join(output_checkpoint_dir, f"model_ft_{train_type}{postfix}.pth")
            print("Selected output checkpoint:", output_checkpoint)
            if not os.path.exists(output_checkpoint):  # Train the model
                print("!! Output checkpoint not found. Training model from scratch...")
                train_model(clean_model, new_train_set_dl, optimizer, criterion, scaler, lr_scheduler,
                            output_checkpoint)
            else:  # Load the model
                print("!! Loading model from pretrained checkpoint:", output_checkpoint)
                clean_model.load_state_dict(torch.load(output_checkpoint, map_location=device))

            # Evaluate the attack success rate for the model trained on clean data
            output_dict = test_unseen_probes(log_predictions, clean_model, device, criterion, test_probes,
                                             val_probe_attacks, tensor_batch_size)

            add_clean_to_output_dict(output_dict, clean_model, device, criterion, test_idx_loader, distributed, rank,
                                     log_predictions)
            output_file = os.path.join(experiment_output_dir, f"attack_success_{train_type}{postfix}.png")
            plot_attack_success_stats(output_dict, label_map_dict, ref_probe_classes, output_file, title=title, log_wandb=log_wandb)
            print("~" * 100)
        print("=" * 100)


def get_last_layer_activations(model, loader, masking_op=None):
    """
        masking_op: Masking operation used on batch images -- used in neural cleanse.

        Returns a tuple of tensors: activations, indices, true classes, predicted classes
            (# examples x activation dim), (# examples), (# examples), (# examples)
    """
    activations = {}

    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()

        return hook

    # TODO: Check that this is correct for non-MNIST
    name, layer = list(model.named_children())[-2]
    handle = layer.register_forward_hook(get_activation(name))

    all_acts, example_indices, classes, class_preds = [], [], [], []

    pbar = tqdm(loader)
    for batch_idx, ((data, target), ex_idx) in enumerate(pbar):
        data = data.to(device)
        if masking_op is not None:
            data = masking_op(data)
        with torch.cuda.amp.autocast(enabled=False):
            output = model(data)
            predictions = torch.argmax(output, 1)

        all_acts.append(activations[name].squeeze())  # Remove size 1 dimensions
        example_indices.append(ex_idx)
        classes.append(target)
        class_preds.append(predictions)

    handle.remove()

    # TODO: SHAPE ISSUES!!!
    return (torch.vstack(all_acts), torch.hstack(example_indices),
            torch.hstack(classes), torch.hstack(class_preds))


def get_confusion_stats(identified_indices, valid_idx, dataset_probe_identity, num_train_probes,
                        verbose=True):
    """
        identified_indices: np array of indices considered to be poisonous by a detector.
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

    per_class = {
        'clean': (num_train_probes * 2 - sum(v for (k, v) in counter.items() if 'clean' in k)) / 2 * num_train_probes,
    }

    for attack in set(dataset_probe_identity) - {'train'}:
        if 'clean' not in attack:
            per_class[attack] = results[attack] / counter[attack]

    if verbose:
        print(f"FPR: {false_positive / (false_positive + true_negative)}")
        print(f"FNR: {false_negative / (false_negative + true_positive)}")
        print("Per class accuracies of detector...")
        for k in per_class:
            print(f"Accuracy ({k}): {per_class[k]}")

    false_positive_rate = false_positive / (false_positive + true_negative)
    true_positive_rate = true_positive / (true_positive + false_negative)
    return false_positive_rate, true_positive_rate


def get_auc(idx_list, valid_idx, dataset_probe_identity, num_train_probes):
    fprs, tprs = [], []

    for identified_indices in idx_list:
        fpr, tpr = get_confusion_stats(identified_indices, valid_idx, dataset_probe_identity,
                                       num_train_probes, verbose=False)
        fprs.append(fpr)
        tprs.append(tpr)

    fprs, tprs = zip(*sorted(zip(fprs, tprs)))
    return sklearn.metrics.auc(fprs, tprs)


def retrain_model(identified_indices, comb_train_indices, idx_dataset, distributed, num_workers, batch_size,
                  num_classes, device, local_rank, lr, momentum, wd, num_epochs, experiment_output_dir,
                  detection_thresh, probes, log_predictions, test_probes, defense):
    print(f"Retraining with {identified_indices.shape[0]} elements removed.")
    retrain_indices = [x for x in comb_train_indices if x not in identified_indices]

    retrain_set_dl = get_loader(idx_dataset, distributed=distributed, num_workers=num_workers,
                                  indices=retrain_indices, batch_size=batch_size)

    clean_model = get_model(dataset, num_classes, device, local_rank, verbose=False)
    clean_criterion, clean_optimizer, clean_lr_scheduler, clean_scaler = \
        get_optimizer(clean_model, device, lr, momentum, wd, num_epochs)

    output_checkpoint_dir = os.path.join(experiment_output_dir, "model_ft")
    if main_proc and not os.path.exists(output_checkpoint_dir):
        os.mkdir(output_checkpoint_dir)
    output_checkpoint = os.path.join(output_checkpoint_dir, f"model_ft_{detection_thresh:.1f}.pth")

    print("Selected output checkpoint:", output_checkpoint)
    if not os.path.exists(output_checkpoint):  # Train the model
        print("!! Output checkpoint not found. Training model from scratch...")
        for epoch in range(num_epochs):
            train(clean_model, device, retrain_set_dl, clean_optimizer, clean_criterion, clean_scaler)
            if (epoch + 1) % 5 == 0:
                print(f"Stats for epoch {epoch + 1}")
                log_results_and_update_stats_and_preds(log_predictions, clean_model, device, clean_criterion, test_idx_loader, distributed,
                                                       rank, new_idx_loader_wo_aug, attack_types, probes, val_probes, defense, tensor_batch_size,
                                                       epoch)  # TODO: Add other args...
            if clean_lr_scheduler is not None:
                clean_lr_scheduler.step()
        torch.save(clean_model.state_dict(), output_checkpoint)
    else:  # Load the model
        print("!! Loading model from pretrained checkpoint:", output_checkpoint)
        clean_model.load_state_dict(torch.load(output_checkpoint, map_location=device))

    # Evaluate accuracy
    print("Retrained model performance:")
    log_results_and_update_stats_and_preds(log_predictions, clean_model, device, clean_criterion, test_idx_loader,
                                           distributed, rank, retrain_set_dl, attack_types, probes, val_probes, defense,
                                           tensor_batch_size, num_epochs+1)
    test_unseen_probes(log_predictions, clean_model, device, clean_criterion, test_probes, val_probe_attacks, tensor_batch_size)
    return clean_model


if defense == "nc":
    def apply_mask_and_trigger(batch, mask, trigger):
        return batch * (1 - mask) + mask * trigger


    def train_cleanse(model, mask, trigger, optimizer, target_class, l1_penalty, train_set,
                      use_autocast=False, log_interval=5):
        optimizer.zero_grad()
        pbar = tqdm(new_idx_loader)
        total_in_cls = 0
        for batch_idx, ((data, target), ex_idx) in enumerate(pbar):
            data = data.to(device)
            triggered_data = apply_mask_and_trigger(data, mask, trigger)
            cleanse_target = torch.full(target.shape, target_class, device='cuda')
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=use_autocast):
                output = model(triggered_data)
                predictions = torch.argmax(output, 1)
                total_in_cls += (predictions == target_class).sum().item()

                trigger_loss = criterion(output, cleanse_target).mean()
                l1_loss = torch.norm(mask, p=1) * l1_penalty
                loss = trigger_loss + l1_loss

            loss.backward()
            optimizer.step()

            mask.data.clamp_(0, 1)
            trigger.data.clamp_(0, 1)

            if batch_idx % log_interval == 0:
                pbar.set_description(f"Loss: {float(loss):.4f}")
            torch.cuda.synchronize()
        print(f"Classifies {total_in_cls} as {target_class}")
        # TODO: This isn't what they did in their paper, but what they did is very fiddly. I suspect this will get
        #   very similar results though.
        if total_in_cls < 0.99 * len(train_set):
            # If not enough are being misclassified, reduce the L1 penalty to allow a larger mask
            l1_penalty /= 2
        if total_in_cls > 0.99 * len(train_set):
            # If more than enough are being misclassified, increase the L1 penalty to keep the mask small.
            l1_penalty *= 2

        return l1_penalty


    def get_indices_for_thresh(fpr_thresh, clean_probe_indices, attacked_classes, poison_acts_by_class,
                               indices_to_check, clean_indices):
        upper_limit = 1 + int(len(clean_probe_indices) * fpr_thresh)
        rejected_indices = []
        for i, atk_class in enumerate(attacked_classes):
            poison_acts = poison_acts_by_class[i]
            # Set a threshold that rejects no more than fpr_thresh of clean probe examples.
            reject_thresh = poison_acts[indices_to_check].sort()[0][-upper_limit]
            rejected_indices.append((poison_acts > reject_thresh).nonzero()[:, 0])
        rejected_indices = torch.hstack(rejected_indices).unique()
        return clean_indices[rejected_indices.cpu()].numpy() if clean_indices is not None else np.array([], dtype=int)

    cleanse_epochs = 20

    masks, norms, triggers = [], [], []

    for cls in range(num_classes):
        l1_penalty = 1.0
        # for every possible label
        mask = torch.nn.Parameter(torch.rand(size=(img_size[-1:] + img_size[:-1]), device='cuda'))
        mask_original = mask.clone()
        trigger = torch.nn.Parameter(torch.rand(size=(img_size[-1:] + img_size[:-1]), device='cuda'))
        cleanse_opt = torch.optim.Adam((mask, trigger))
        for _ in range(cleanse_epochs):
            l1_penalty = train_cleanse(model, mask, trigger, cleanse_opt, cls, l1_penalty, train_set)

        masks.append(mask.detach())
        triggers.append(trigger.detach())
        norms.append(torch.norm(mask, p=1))
        print(f"Trained mask for class {cls}. Final l1 penalty: {l1_penalty}, final mask magnitude: {mask.norm(p=1)}")

    # Calculate MAD
    torch.tensor(norms).median()
    norms = torch.tensor(norms)
    median = norms.median()
    absolute_deviations = (norms - median).abs()
    mad = 1.4826 * absolute_deviations.median()  # magic number from paper +
                                                 #  https://en.wikipedia.org/wiki/Median_absolute_deviation
    anomaly_index = absolute_deviations / mad

    attacked_classes = (anomaly_index >= 2).nonzero()[:, 0]  # threshold 2 recommended in paper

    # from paper -- take top 1% of neurons by diff in activations
    adv_neuron_thresh = 0.01
    # again, from paper. Can adjust? TODO: maybe need to adjust for multiple attack classes
    print("Skipping detected attacks where norm is above median.")
    attacked_classes = attacked_classes[(norms[attacked_classes] <= median).nonzero()[:, 0]]

    clean_activations, clean_indices, _, _ = get_last_layer_activations(model, new_idx_loader)
    clean_probe_indices = np.array(train_probe['clean_idx'])
    indices_to_check = torch.isin(clean_indices, torch.tensor(clean_probe_indices)).nonzero()[:, 0]

    poison_acts_by_class = []

    for atk_class in attacked_classes:
        # Get all activations
        mask, trigger = masks[atk_class], triggers[atk_class]
        dirty_activations, dirty_indices, _, dirty_predictions = get_last_layer_activations(model, new_idx_loader,
                                                                            masking_op=lambda img: apply_mask_and_trigger(img, mask, trigger))
        # See how successful the attacks were
        attack_success = (dirty_predictions == atk_class).sum()
        print(f"Anomalous class {atk_class.item()} with index {anomaly_index[atk_class].item():.3f} "
              f"has mask magnitude {norms[atk_class].item():.3f} which classifies {attack_success.item()} as {atk_class.item()}")
        # Reorder the activations (train loader doesn't guarantee order)
        clean_order = clean_indices.argsort()
        clean_activations, clean_indices = clean_activations[clean_order], clean_indices[clean_order]
        dirty_order = dirty_indices.argsort()
        dirty_activations, dirty_indices = dirty_activations[dirty_order], dirty_indices[dirty_order]

        # per example differences
        mean_activation_diff = (dirty_activations - clean_activations).mean(axis=0)
        neurons_to_select = int(len(mean_activation_diff) * adv_neuron_thresh)
        poisoned_neurons = mean_activation_diff.sort()[1][-neurons_to_select:]  # Select indices of poisoned neurons

        # Activations of poisoned neurons on clean images
        #    if this is high, then the image itself likely carries the poison.
        acts_of_poisoned = clean_activations[:, poisoned_neurons].mean(axis=1)
        poison_acts_by_class.append(acts_of_poisoned)

    fpr_thresh = 0.05 / max(len(attacked_classes), 1)
    # Calculate AUC
    auc_fpr_threshes = np.geomspace(fpr_thresh / 20, 0.8, num=50)

    auc_idx = []
    for auc_fpr_threshes in auc_fpr_threshes:
        auc_idx.append(get_indices_for_thresh(auc_fpr_threshes, clean_probe_indices, attacked_classes,
                                              poison_acts_by_class, indices_to_check, clean_indices))
    print("NC AUC", get_auc(auc_idx, valid_idx, dataset_probe_identity, num_train_probes))

    # Retrain model using default threshold
    identified_indices = get_indices_for_thresh(fpr_thresh, clean_probe_indices, attacked_classes,
                                                poison_acts_by_class, indices_to_check, clean_indices)

    # Print confusion stats...
    get_confusion_stats(identified_indices, valid_idx, dataset_probe_identity, num_train_probes)
    clean_model = retrain_model(identified_indices, comb_train_indices, idx_dataset, distributed, num_workers,
                                batch_size, num_classes, device, local_rank, lr, momentum, wd, num_epochs,
                                experiment_output_dir, fpr_thresh, probes, log_predictions, test_probes, defense)

    # fpr_thresh is only misnamed parameter...
    print("Done with nc")
    # TODO: retrain, repeat...
    # This doesn't work because the norms of the selected classes are higher than the median! Can rule these out,
    #   but that would mean that, by default, NC would pick up on nothing!!
    # Solution... somehow check the difference between high and lows?? This is a substantial extension...


if defense == "ac":
    def get_indices_for_class_clusters(detected_classes, clusterings, activation_by_predicted_class):
        identified_indices = []

        for cls in detected_classes:
            clustering = clusterings[cls]
            # If more 1s than 0s, get indices of 0s
            if sum(clustering) * 2 > len(clustering):
                selected_indices = np.where(clustering == 0)
            else:
                selected_indices = np.where(clustering == 1)
            dataset_indices = activation_by_predicted_class[cls][1][selected_indices]
            identified_indices.append(dataset_indices)

        return torch.hstack(identified_indices) if identified_indices else np.array([], dtype=int)

    activations, indices, classes, predictions = get_last_layer_activations(model, new_idx_loader)
    activation_by_predicted_class = {}
    for i in range(num_classes):
        indices_to_pick = np.where(predictions.cpu() == i)[0]
        activation_by_predicted_class[i] = (activations[indices_to_pick], indices[indices_to_pick])

    dim_reducer = sklearn.decomposition.FastICA(n_components=10)  # Magic number from paper
    clusterer = sklearn.cluster.KMeans(n_clusters=2)  # Unclear if this can be reasonably extended -- are there clustering algos that learn the number of clusters?

    clusterings = []
    rsc_scores = []
    sil_scores = []
    for cls in range(num_classes):
        data = activation_by_predicted_class[cls][0].cpu()
        data = data[:, data.sum(dim=0).bool()]  # remove zero columns, otherwise dim reduction outputs all 0s

        fit = dim_reducer.fit_transform(data)
        clustering = clusterer.fit_predict(fit)
        # Relative size comparison
        rsc_score = sum(clustering) / len(data)
        if rsc_score > 0.5:
            rsc_score = 1 - rsc_score
        sil_score = sklearn.metrics.silhouette_score(fit, clustering)

        clusterings.append(clustering)
        rsc_scores.append(rsc_score)
        sil_scores.append(sil_score)

    # TODO: Use ExRe? Seems like there's too much of a cost in terms of computation...
    # Use this to select which score to reclassify with.
    mode = "sil"
    if mode == "sil":
        detect_thresh = 0.15  # Less aggressive -- 0.1 would be more aggressive.
        detected_classes = [i for i in range(num_classes) if sil_scores[i] > detect_thresh]

        auc_detect_threshes = np.geomspace(0.01, 0.5, num=50)
    elif mode == "rsc":
        detect_thresh = 0.3  # Worst case from AC
        detected_classes = [i for i in range(num_classes) if rsc_scores[i] < detect_thresh]
        auc_detect_threshes = np.geomspace(0.01, 0.4, num=50)
    else:
        raise NotImplementedError

    # Calculate AUC
    auc_idx = []
    for detect_thresh in auc_detect_threshes:
        if mode == "sil":
            auc_detected_classes = [i for i in range(num_classes) if sil_scores[i] > detect_thresh]
        elif mode == "rsc":
            auc_detected_classes = [i for i in range(num_classes) if rsc_scores[i] < detect_thresh]
        auc_idx.append(get_indices_for_class_clusters(auc_detected_classes, clusterings, activation_by_predicted_class))

    print("AC AUC", get_auc(auc_idx, valid_idx, dataset_probe_identity, num_train_probes))

    # Retrain model using default threshold
    identified_indices = get_indices_for_class_clusters(detected_classes, clusterings, activation_by_predicted_class)
    get_confusion_stats(identified_indices, valid_idx, dataset_probe_identity, num_train_probes)
    clean_model = retrain_model(identified_indices, comb_train_indices, idx_dataset, distributed, num_workers,
                                batch_size, num_classes, device, local_rank, lr, momentum, wd, num_epochs,
                                experiment_output_dir, detect_thresh, probes, log_predictions, test_probes,
                                defense)  # detect_thresh...

    # Use probes to get expected clean silhouette scores per class
    # Get silhouette score -- if it's far from clean, then mark the smaller cluster as dirty. Repeat through all classes.
    # Ex-Re? It seems too difficult to do Ex-Re 20x training runs for each of 10 classes,
    #   then do AC all over again on the retrained model to find the second backdoor...

if defense == "ss":
    # For every class:
    #   n = # training examples labeled y
    #   R_hat = average d-dim representation (at last layer) of train examples
    #   M = [R(x) - R_hat] = n x d matrix of centered representation
    #   v = top right singular vector of M
    #   tau = ([R(x_i) - R_hat] * v)^2 for all i (n dimensional)
    #   Remove the top 1.5*epsilon (thresholding value) from dataset.
    # Retrain
    def get_indices_for_eps(eps_thresh, num_classes, taus, cls_idx, indices):
        rejected_indices = []
        for cls in range(num_classes):
            tau = taus[cls]
            cls_indices = cls_idx[cls]
            num_to_remove = int(len(tau) * eps_thresh * 1.5)
            rejected_indices.append(cls_indices[tau.argsort()[-num_to_remove:].cpu()])

        rejected_indices = torch.hstack(rejected_indices).unique()
        return indices[rejected_indices.cpu()].numpy()

    activations, indices, classes, predictions = get_last_layer_activations(model, new_idx_loader)
    taus, cls_idx = [], []
    for cls in range(num_classes):
        cls_indices = (classes == cls).nonzero()[:, 0]
        cls_activations = activations[cls_indices]
        m = cls_activations - cls_activations.mean(axis=0)
        u, s, v = m.svd()
        v_top = v[:, 0]  # TODO: Check this
        tau = m.matmul(v_top) ** 2

        taus.append(tau)
        cls_idx.append(cls_indices)

    epsilon_thresh = 0.1  # From paper, assuming 10% poisoning max

    # Calculate AUC
    auc_eps_threshes = np.geomspace(0.01, 0.5, num=50)
    auc_idx = []
    for auc_eps_thresh in auc_eps_threshes:
        auc_idx.append(get_indices_for_eps(auc_eps_thresh, num_classes, taus, cls_idx, indices))

    print("SS AUC", get_auc(auc_idx, valid_idx, dataset_probe_identity, num_train_probes))

    # Retrain model using default threshold
    identified_indices = get_indices_for_eps(epsilon_thresh, num_classes, taus, cls_idx, indices)

    # Print confusion stats...
    get_confusion_stats(identified_indices, valid_idx, dataset_probe_identity, num_train_probes)
    clean_model = retrain_model(identified_indices, comb_train_indices, idx_dataset, distributed, num_workers,
                                batch_size, num_classes, device, local_rank, lr, momentum, wd, num_epochs,
                                experiment_output_dir, epsilon_thresh, probes, log_predictions, test_probes, defense)

    print("Done with ss")
    # TODO: retrain, repeat...


if defense == "freq":
    # They use a series of transformations similar to backdoor images to train their detector.
    # We could do that to reproduce their results exactly:
    #   * White square
    #   * Colored noise square
    #   * Gaussian noise
    #   * Random shadow
    #   * Random blend
    # However, I think it's probably more fair to mapd to do an identical comparison -- train their detector on
    #   bona fide clean and inserted backdoor probe examples, the same way mapd is trained on bona fide clean and
    #   backdoor probes examples.
    freq_probes = {
        'clean': train_probe['clean'].cpu().numpy(),
        'backdoor': train_probe['backdoor'].cpu().numpy(),
    }

    def dct2(block):
        # Copied from:
        #   https://github.com/YiZeng623/frequency-backdoor/blob/main/Sec4_Frequency_Detection/Train_Detection.ipynb
        return dct(dct(block.T, norm='ortho').T, norm='ortho')


    def get_indices_for_thresh_from_loader(thresh, new_idx_loader_wo_aug, freq_model):
        freq_model.eval()
        identified_indices = []
        for (image, label), indices in tqdm(new_idx_loader_wo_aug):
            image = image.cpu().numpy()
            num_images = image.shape[0]
            channels = image.shape[1]  # NCHW required
            for n in range(num_images):
                for c in range(channels):
                    image[n, c, :, :] = dct2(image[n, c, :, :])

            image = torch.tensor(image, device=device)
            outputs = freq_model(image)
            outputs = torch.nn.functional.softmax(outputs, dim=1)

            probs = outputs[:, 1]

            identified_indices.append(indices[(probs.cpu() >= thresh).nonzero()[:, 0]])

        return torch.hstack(identified_indices)


    for key in freq_probes:
        num_images = freq_probes[key].shape[0]
        channels = freq_probes[key].shape[1]  # NCHW required
        for n in range(num_images):
            for c in range(channels):
                freq_probes[key][n, c, :, :] = dct2(freq_probes[key][n, c, :, :])

    freq_train_set = torch.vstack((torch.tensor(freq_probes['clean']),
                                   torch.tensor(freq_probes['backdoor'])))
    freq_labels = torch.hstack((torch.zeros(freq_probes['clean'].shape[0], dtype=torch.long),
                                torch.ones(freq_probes['backdoor'].shape[0], dtype=torch.long)))

    freq_dataset = TensorDataset(freq_train_set, freq_labels)
    freq_dataloader = DataLoader(freq_dataset, batch_size=32, shuffle=True)

    freq_model = FreqCNN(freq_train_set[0].shape).to(device)

    freq_criterion = torch.nn.CrossEntropyLoss()
    freq_optimizer = torch.optim.Adadelta(freq_model.parameters(), lr=0.05, weight_decay=1e-4)

    model.train()
    for epoch in range(10):
        epoch_loss = 0.
        epoch_correct = 0
        for batch, labels in freq_dataloader:
            batch, labels = batch.to(device), labels.to(device)

            freq_optimizer.zero_grad()

            outputs = freq_model(batch)

            correct = (outputs.argmax(axis=1) == labels).sum()
            loss = freq_criterion(outputs, labels)
            loss = loss.sum()
            loss.backward()
            freq_optimizer.step()

            epoch_loss += loss.item()
            epoch_correct += correct.item()
        print(f"Epoch {epoch+1} loss: {epoch_loss/len(freq_dataset):.6f}, acc: {epoch_correct/len(freq_dataset):.6f}")

    detection_thresh = 0.5

    # Calculate AUC
    auc_detect_threshes = np.linspace(0.1, 0.9, num=9)
    auc_idx = []
    for auc_detect_thresh in auc_detect_threshes:
        auc_idx.append(get_indices_for_thresh_from_loader(auc_detect_thresh, new_idx_loader_wo_aug, freq_model))

    print("Freq AUC", get_auc(auc_idx, valid_idx, dataset_probe_identity, num_train_probes))

    # Retrain model using default threshold
    identified_indices = get_indices_for_thresh_from_loader(detection_thresh, new_idx_loader_wo_aug, freq_model)
    get_confusion_stats(identified_indices, valid_idx, dataset_probe_identity, num_train_probes)
    retrain_model(identified_indices, comb_train_indices, idx_dataset, distributed, num_workers, batch_size,
                  num_classes, device, local_rank, lr, momentum, wd, num_epochs, experiment_output_dir,
                  detection_thresh, probes, log_predictions, test_probes, defense)

if defense == "abl":
    def get_indices_from_losses(thresh, train_set, loss_idx, ex_idx):
        num_ex_unlearning = int(len(train_set) * thresh)
        identified_indices = loss_idx[:num_ex_unlearning]
        identified_indices = [int(ex_idx[i]) for i in identified_indices]
        return np.array(identified_indices)


    num_pretrain_epochs = 10
    flooding_threshold = 0.5

    model = get_model(dataset, num_classes, device, local_rank, verbose=False)

    finetune_and_unlearn = False  # If this is true, then normal ABL is done.
    if finetune_and_unlearn:
        selection_threshold = 0.01  # 1% of the total examples, even though the poisoning ratio is 10%
    else:
        selection_threshold = 0.15  # Comparable to spectral signatures...

    # Step # 01: Regular pretraining
    print("!! Performing initial pretraining with all examples (using loss flooding)...")
    output_checkpoint_file = os.path.join(experiment_output_dir, "model_pretrain.pth")
    if not os.path.exists(output_checkpoint_file):
        criterion, optimizer, lr_scheduler, scaler = get_optimizer(model, device, lr, momentum, wd, num_pretrain_epochs)
        for epoch in tqdm(range(num_pretrain_epochs)):
            train(model, device, new_idx_loader, optimizer, criterion, scaler, flooding_threshold=flooding_threshold)
            if epoch % 5 == 4:
                log_results_and_update_stats_and_preds(log_predictions, model, device, criterion, test_idx_loader,
                                                       distributed, rank, new_idx_loader_wo_aug, attack_types, probes,
                                                       val_probes, defense, tensor_batch_size, epoch)
        torch.save(model.state_dict(), output_checkpoint_file)
    else:
        print(f"!! Loading pretrained checkpoint file:", output_checkpoint_file)
        model.load_state_dict(torch.load(output_checkpoint_file, map_location=device))


    # Step # 02: identify backdoored examples based on the loss value
    # Get the loss values for all examples in the dataset
    _, pred_output_dict = test(model, device, criterion, new_idx_loader, distributed, rank, log_predictions=True)
    losses = pred_output_dict["loss"]
    ex_idx = pred_output_dict["ex_idx"]

    loss_idx = np.argsort(losses)  # Ascending sort

    # Calculate AUC
    auc_threshes = np.geomspace(0.001, 0.5, num=50)
    auc_idx = []
    for auc_thresh in auc_threshes:
        auc_idx.append(get_indices_from_losses(auc_thresh, train_set, loss_idx, ex_idx))
    print("ABL AUC", get_auc(auc_idx, valid_idx, dataset_probe_identity, num_train_probes))

    # Retrain model using default threshold
    indices_to_maximize = get_indices_from_losses(selection_threshold, train_set, loss_idx, ex_idx)

    # Identify the maximum indices
    total_ex = len(new_idx_loader.dataset)
    missing_vals = [x for x in range(total_ex) if x not in ex_idx]
    print(
        f"Ex idx stats / # vals: {total_ex} / Total: {len(ex_idx)} / Unique vals: {len(np.unique(ex_idx))} / Missing vals: {len(missing_vals)} / max idx: {np.max(ex_idx)}")
    print(f"Indices to maximize / Len: {len(indices_to_maximize)} / Indices: {indices_to_maximize}")

    ground_truth_probes = [i for i in range(len(ex_idx), total_ex)]
    backdoor_probe_idx = [i for i in ground_truth_probes if "clean" not in dataset_probe_identity[i]]

    backdoor_probes_detected = [i for i in indices_to_maximize if i in backdoor_probe_idx]
    print(f"Backdoor probe examples flagged: {len(backdoor_probes_detected)} / {backdoor_probes_detected}")
    print(f"Ground truth probes: {len(ground_truth_probes)} / {ground_truth_probes}")

    if not finetune_and_unlearn:
        identified_indices = np.array(indices_to_maximize)
        # Retrain with selected indices like normal
        get_confusion_stats(identified_indices, valid_idx, dataset_probe_identity, num_train_probes)
        retrain_model(identified_indices, comb_train_indices, idx_dataset, distributed, num_workers, batch_size,
                      num_classes, device, local_rank, lr, momentum, wd, num_epochs, experiment_output_dir,
                      selection_threshold, probes, log_predictions, test_probes, defense)

    else:  # Finetune and unlearn
        clean_finetuning_epochs = 60
        unlearning_epochs = 5
        use_gt_backdoors = True  # Note: this is important to set.

        if use_gt_backdoors:
            print(f"[WARNING] Using the ground-truth backdoors for anti-backdoor learning")
            indices_to_maximize = backdoor_probe_idx[:int(len(train_set) * selection_threshold)]
            print(f"Total probe idx: {len(ground_truth_probes)} / Backdoor idx: {len(backdoor_probe_idx)}")
        remaining_indices = [i for i in range(total_ex) if i not in indices_to_maximize and i not in missing_vals]
        print(
            f"Selected indices / Clean indices: {len(remaining_indices)} / Backdoored indices: {len(indices_to_maximize)}")

        # Step # 03: generate dataloaders based on the clean and backdoor indices
        clean_dl = get_loader(new_idx_loader.dataset, distributed=distributed, indices=remaining_indices,
                              num_workers=num_workers, batch_size=batch_size)
        detected_backdoors_dl = get_loader(new_idx_loader.dataset, distributed=distributed, indices=indices_to_maximize,
                                           num_workers=num_workers, batch_size=batch_size)

        # Step # 04: finetune the model only on clean data
        print("!! Starting finetuning phase on the clean examples...")
        model_postfix = "_gt_backdoors" if use_gt_backdoors else ""
        output_checkpoint_file = os.path.join(experiment_output_dir, f"model_clean_ft{model_postfix}.pth")
        if not os.path.exists(output_checkpoint_file):
            lr = 0.1
            criterion, optimizer, lr_scheduler, scaler = get_optimizer(model, device, lr, momentum, wd,
                                                                       clean_finetuning_epochs)
            for epoch in tqdm(range(clean_finetuning_epochs)):
                output_dict = train(model, device, clean_dl, optimizer, criterion, scaler)
                if epoch % 5 == 4:
                    log_results_and_update_stats_and_preds(log_predictions, model, device, criterion, test_idx_loader,
                                                           distributed, rank, new_idx_loader_wo_aug, attack_types, probes,
                                                           val_probes, defense, tensor_batch_size, epoch)
            torch.save(model.state_dict(), output_checkpoint_file)
        else:
            print(f"!! Loading clean finetuned checkpoint file:", output_checkpoint_file)
            model.load_state_dict(torch.load(output_checkpoint_file, map_location=device))

        # Step # 05: perform unlearning step on the identified backdoored examples
        print("!! Starting unlearning phase on the identified backdoor examples...")
        output_checkpoint_file = os.path.join(experiment_output_dir, f"model_unlearned{model_postfix}.pth")
        if not os.path.exists(output_checkpoint_file):
            lr = 5e-4
            criterion, optimizer, lr_scheduler, scaler = get_optimizer(model, device, lr, momentum, wd, unlearning_epochs)
            for epoch in tqdm(range(unlearning_epochs)):
                output_dict = train(model, device, detected_backdoors_dl, optimizer, criterion, scaler,
                                    gradient_ascent=True)
                if epoch % 5 == 4:
                    log_results_and_update_stats_and_preds(log_predictions, model, device, criterion, test_idx_loader,
                                                           distributed, rank, new_idx_loader_wo_aug, attack_types, probes,
                                                           val_probes, defense, tensor_batch_size, epoch)
            torch.save(model.state_dict(), output_checkpoint_file)
        else:
            print(f"!! Loading unlearned checkpoint file:", output_checkpoint_file)
            model.load_state_dict(torch.load(output_checkpoint_file, map_location=device))

        test_stats, test_preds = test(model, device, criterion, test_idx_loader, distributed, rank,
                                      log_predictions=log_predictions)
        test_unseen_probes(log_predictions, model, device, criterion, test_probes, val_probe_attacks, tensor_batch_size)
