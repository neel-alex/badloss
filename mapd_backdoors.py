#!/usr/bin/env python
import os
import copy
import json
import pickle
import subprocess
import shutil
import warnings
from tqdm import tqdm
from collections import Counter


import wandb
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import sklearn.neighbors
import sklearn.cluster
import sklearn.decomposition
import sklearn.metrics
from sklearn.metrics import roc_curve, auc
from scipy.fftpack import dct


import config
import utils
import dist_utils
from dataset_utils import get_settings_for_dataset, make_probe_dataset, \
                          make_index_dataset, get_loader, IdxDataset
from plot_utils import plot_probe_examples
from backdoors import make_train_probes, make_val_probes, make_test_probes
from torch_utils import get_model, get_optimizer, train, test, test_tensor, \
    FreqCNN, collect_losses, train_cbd, train_intraclass, calc_fct, pss_unlearn


from cognitive_distillation import CognitiveDistillation
from cbd_util import DisenEstimator


# Required for determinism
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
torch.use_deterministic_algorithms(True, warn_only=True)


args = config.config().parse_args()
print(args.dataset, args.attack, args.defense, args.poisoning_ratio)

# TODO
attacks = config.get_attacks(args.attack, args.dataset)
poison_ratios = config.get_poisoning_ratio(args.dataset, args.poisoning_ratio)

# Set random seed
seed = args.seed
print("Seed:", seed)
utils.seed_all(seed)

# Essential config
log_predictions = True
project_id = "exp68"
experiment_output_dir = f"./backdoor_{project_id}_{args.dataset}_{args.defense}_{args.attack}{'_' + str(args.poisoning_ratio) if args.poisoning_ratio is not [] else ''}"
model_collection_dir = f"./backdoor_{project_id}_model_{args.dataset}_{args.attack}{'_' + args.defense if args.defense in {'badloss'} else ''}{'_' + str(args.poisoning_ratio) if args.poisoning_ratio is not [] else ''}"
# model_collection_dir = experiment_output_dir
num_workers = 8  # TODO: Warning that the number of workers requested isn't right?
surface_examples = False

# Initalize W&B (opt-in) -- assumes wandb is already logged in
log_wandb = False
if args.wandb and dist_utils.is_main_proc():
    print("Initializing w&b")
    wandb_project = f"mapd_backdoors_{args.dataset}"
    wandb_run_name = f"attack_{args.attack}_defense_{args.defense}{'_poisoning_ratio' + str(args.poisoning_ratio) if args.poisoning_ratio is not None else ''}_run_{project_id}"
    wandb.init(
        project=wandb_project,
        name=wandb_run_name,
    )
    log_wandb = True

# Initialize the distributed environment
distributed = False
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


def _git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       cwd=os.path.dirname(os.path.abspath(__file__))).strip()
    except Exception:
        return None


def _jsonable(x):
    if torch.is_tensor(x):
        return x.tolist()
    if isinstance(x, (np.generic, np.ndarray)):
        return x.tolist()
    raise TypeError(f"Not JSON serializable: {type(x)}")


results = {"args": vars(args), "git_sha": _git_sha()}


def save_results():
    if main_proc:
        with open(os.path.join(experiment_output_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2, default=_jsonable)


def record_model_metrics(key, test_stats, asr_dict):
    results[key] = {"clean_acc": test_stats["acc"], "asr": {k: v["accuracy"] for k, v in asr_dict.items()}}
    save_results()


def record_detection(**kwargs):
    results.setdefault("detection", {}).update(kwargs)
    save_results()


# Shunting off this logic to another file -- TODO select file location arg?
(img_size, train_transform, test_transform, no_transform,
 data_dir, train_set, train_set_wo_aug, test_set) = get_settings_for_dataset(args.dataset)


print(args.dataset, len(train_set), len(test_set))

# ## Setup probes

num_classes = len(train_set.classes)
if args.dataset in ["mnist", "cifar10", "imagenette"]:
    assert num_classes == 10
elif args.dataset == "cifar100":
    assert num_classes == 100
elif args.dataset == "gtsrb":
    assert num_classes == 43
else:
    assert num_classes == 1000
print(args.dataset, num_classes)


# Standardizing nomenclature...
if args.defense == "badloss":
    attack_types = ["backdoor"] + [f"backdoor_{attack}" for attack in attacks]
else:
    attack_types = [f"backdoor_{attack}" for attack in attacks]

val_probes, attack_targets, random_pattern, warping_grids = make_val_probes(num_classes, args.dataset, train_set_wo_aug,
                                                                            poison_ratios, attacks,
                                                                            experiment_output_dir, main_proc, img_size,
                                                                            device,)

include_val_probe_examples = False

train_probe_attack = 'clean'
train_probe, attack_target, aux_data = make_train_probes(num_classes, args.dataset, train_set_wo_aug,
                                                         args.num_train_probes, train_probe_attack,
                                                         experiment_output_dir, main_proc, img_size, device,
                                                         val_probe_indices=val_probes["all_backdoor_idx"],
                                                         include_val_probe_examples=include_val_probe_examples)

train_probes_idx = train_probe["all_backdoor_idx"]

num_test_probes = 10000
test_probes = make_test_probes(test_set, args.dataset, num_test_probes, attacks, attack_targets,
                               random_pattern, warping_grids, experiment_output_dir, main_proc, img_size, device)


# Merge probe dicts
if args.defense == "badloss":
    probes = {**train_probe, **val_probes}
    unified_backdoor_idx = np.concatenate((train_probe['all_backdoor_idx'], val_probes['all_backdoor_idx']))
    probes['all_backdoor_idx'] = unified_backdoor_idx
    chosen_attack_targets = {**{'backdoor': attack_target}, **attack_targets}
    plot_probe_examples(probes, args.dataset, train_set, attack_types, rank, experiment_output_dir, log_wandb=log_wandb)
else:
    probes = val_probes


num_epochs = args.num_epochs if args.num_epochs is not None \
                else config.get_num_epochs(args.dataset)
batch_size = args.batch_size

tensor_batch_size = batch_size if args.dataset in {"gtsrb", "imagenette", "imagenet"} else 128
lr = 0.1
momentum = 0.9
wd = 0.0001
moving_avg_weight = None
augment_in_pretraining = False if args.defense == "badloss" else True
augment_in_retraining = True
if augment_in_retraining == False:
    raise NotImplementedError("Set augment = False in dataset_utils instead, good luck.")


comb_train_set, comb_train_indices, dataset_probe_identity, discarded_idx = \
    make_probe_dataset(probes, train_set, args.dataset, args.num_train_probes, args.defense,
                       train_transform, attacks, experiment_output_dir,
                       device, include_val_probe_examples=include_val_probe_examples)
valid_idx = [i for i in range(len(train_set)) if i not in discarded_idx]

model = get_model(args.dataset, num_classes, device, local_rank, verbose=False, arch=args.arch)
criterion, optimizer, lr_scheduler, scaler = get_optimizer(model, device, lr, momentum, wd, num_epochs)

new_idx_loader, new_idx_loader_wo_aug, test_idx_loader, idx_dataset = \
    make_index_dataset(comb_train_set, comb_train_indices, test_set,
                       no_transform, batch_size, distributed, num_workers, seed)

# Load train probes onto gpu -- inexpensive and saves time.
# ...could load entire train set onto gpu (15GB tops in GTSRB), but that's a pain, code-wise.
to_cuda = copy.deepcopy(attack_types)
if args.defense == 'badloss':
    if include_val_probe_examples:
        to_cuda += ['backdoor_val', 'clean', 'clean_val']
    else:
        to_cuda += ['clean']

for key in to_cuda:
    probes[key] = probes[key].to(device)
    probes[f'{key}_labels'] = probes[f'{key}_labels'].to(device)
# Non-GTSRB test probes can stay on the GPU
if args.dataset != "gtsrb" and args.dataset != "imagenette" and args.dataset != "imagenet":
    for key in test_probes:
        test_probes[key] = test_probes[key].to(device)


model_dir = os.path.join(model_collection_dir, f"models_{args.dataset}")
model_file = os.path.join(model_dir, f"model_{args.dataset}_dynamics.pth")
data_file = os.path.join(model_collection_dir, f"stats_{args.dataset}_dynamics.pkl")
data_statistics_file = os.path.join(model_collection_dir, f"stats_{args.dataset}_data_statistics.pkl")


# In[ ]:


if main_proc and not os.path.exists(model_dir):
    os.mkdir(model_dir)


# In[ ]:


ref_probe_classes = ["backdoor", "clean"]

# In[ ]:

def log_results_and_update_stats_and_preds(log_predictions, model, device, criterion, test_idx_loader, distributed,
                                           rank, new_idx_loader_wo_aug, attack_types, probes, val_probes, defense,
                                           tensor_batch_size, epoch, statistics=None, predictions=None, use_eval_mode=True,
                                           max_loss_val_bound=None):
    test_stats, test_preds = test(model, device, criterion, test_idx_loader, distributed, rank,
                                  log_predictions=log_predictions)
    ret_test_stats = test_stats
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
    if defense == "badloss":
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
    return ret_test_stats


def test_unseen_probes(log_predictions, model, device, criterion, test_probes, attacks, tensor_batch_size):
    output_dict = {}
    all_attacks = [x for x in test_probes.keys() if not x.endswith("_labels")]
    for attack in all_attacks:
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
if args.defense in {"nc", "ac", "ss", "cd", "pss"}:
    if not os.path.exists(model_file):
        loader = new_idx_loader if augment_in_pretraining else new_idx_loader_wo_aug
        for epoch in range(num_epochs):
            train(model, device, loader, optimizer, criterion, scaler)
            if (epoch + 1) % 5 == 0:
                print(f"Stats for epoch {epoch + 1}")
                log_results_and_update_stats_and_preds(log_predictions, model, device, criterion, test_idx_loader,
                                                       distributed, rank, new_idx_loader_wo_aug, attack_types, probes,
                                                       val_probes, args.defense, tensor_batch_size, epoch)
                test_unseen_probes(log_predictions, model, device, criterion, test_probes, attacks,
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
        print("Data files already found. Loading data from saved checkpoints.")

        model.load_state_dict(torch.load(model_file, map_location=device))
elif args.defense == "badloss":
    stats = {}
    # TODO: revisit names
    model_file = os.path.join(model_dir, f"model_{args.dataset}.pth")
    data_file = os.path.join(model_collection_dir, f"stats_{args.dataset}.pkl")

    if os.path.exists(model_file):
        assert os.path.exists(data_file)
        print("Data files already found. Loading data from saved checkpoints.")
        model.load_state_dict(torch.load(model_file, map_location=device))
        assert os.path.exists(data_file)
        with open(data_file, "rb") as f:
            stats = pickle.load(f)

    else:
        losses = torch.zeros((len(new_idx_loader.dataset), args.badloss_pretrain_epochs))
        correct_class_probs = torch.zeros((len(new_idx_loader.dataset), args.badloss_pretrain_epochs))

        collect_losses_in_training = False  # TODO: Implement
        use_eval_mode = True  # eval mode BN
        loader = new_idx_loader if augment_in_pretraining else new_idx_loader_wo_aug
        
        for epoch in range(args.badloss_pretrain_epochs):
            output_dict = train(model, device, loader, optimizer, criterion, scaler) # TODO
            if not collect_losses_in_training:
                loss_array, probs_array = collect_losses(model, device, new_idx_loader_wo_aug, criterion, scaler)
                losses[:, epoch] = loss_array
                correct_class_probs[:, epoch] = probs_array

            if epoch % 5 == 4:
                test_unseen_probes(log_predictions, model, device, criterion, test_probes, attacks,
                                    tensor_batch_size)
            if lr_scheduler is not None:
                lr_scheduler.step()

        test_unseen_probes(log_predictions, model, device, criterion, test_probes, attacks, tensor_batch_size)
        stats['losses'] = losses
        stats['probs'] = correct_class_probs
        stats['probe_id'] = dataset_probe_identity

        if main_proc:
            # Save the model
            torch.save(model.state_dict(), model_file)

            # Save the final data
            with open(data_file, "wb") as f:
                pickle.dump(stats, f, protocol=pickle.HIGHEST_PROTOCOL)
elif args.defense in {"freq", "abl", "cbd"}:
    pass
else:
    raise NotImplementedError

print("Final model performance:")
attacked_test_stats, _ = test(model, device, criterion, test_idx_loader, distributed, rank, log_predictions=log_predictions)
attacked_asr = test_unseen_probes(log_predictions, model, device, criterion, test_probes, attacks, tensor_batch_size)
# NB: for freq/abl/cbd this is the untrained initial model (those defenses train their own)
record_model_metrics("attacked", attacked_test_stats, attacked_asr)


if args.defense == "badloss":
    if args.badloss_metric == 'loss':
        losses = stats['losses']
    elif args.badloss_metric == 'prob':
        losses = stats['probs']  # TODO: refactor
    else:
        raise NotImplementedError

    dataset_probe_identity = np.array(dataset_probe_identity)
    clean_idx = np.where(dataset_probe_identity == 'clean')[0]

    print("Training the trajectory classifier...")

    # Create the one-class classifier
    clean_trajectories = losses[clean_idx]
    clean_labels = np.array([0 for _ in range(len(clean_trajectories))])
    oc_clf_neighbors = len(clean_trajectories)  # TODO: Change to lower number?
    oc_clf = sklearn.neighbors.KNeighborsClassifier(oc_clf_neighbors)
    oc_clf.fit(clean_trajectories, clean_labels)

    # Evaluate AUCs
    # TODO...

    # TODO: Refactor this
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

    # TODO: Move to config
    take_weighted_average = True
    use_one_class = True
    use_exp_weighting = True
    use_filtering = True
    oc_eps = 0.01
    n_neighbors = 50
    take_weighted_average = False

    print("!! Collecting clean training indices...")
    missing_vals = (losses == 0).all(dim=1)  # Identify probe examples
    missing_vals_idx = torch.nonzero(missing_vals).squeeze()
    available_ex = ~missing_vals
    num_missing_vals = sum(missing_vals)
    print(f"!! Total loss traj len: {len(losses)} / Modified examples identified: {num_missing_vals}")
    
    fit_data = oc_clf._fit_X
    # TODO: Do this earlier?
    print("using avail")
    epoch_avg_loss = losses[available_ex].mean(axis=0)  # TODO: clean_idx or available_ex?
    final_avgs = epoch_avg_loss[:3].tolist()
    epochs_to_keep = [0, 1, 2]
    for i in range(3, len(epoch_avg_loss)):
        if epoch_avg_loss[i] < 2 * sum(final_avgs[-3:])/3: # If less than twice the average of the previous 3 VALID losses
            epochs_to_keep.append(i)
            final_avgs.append(epoch_avg_loss[i])

    print(epochs_to_keep, final_avgs)
    filtered_fit_data = fit_data[:, epochs_to_keep]
    filtered_losses_np = losses[:, epochs_to_keep]
    filtered_clf = sklearn.neighbors.KNeighborsClassifier(oc_clf.n_neighbors)
    filtered_clf.fit(filtered_fit_data, clean_labels)

    if use_filtering:
        losses_np = filtered_losses_np
        clf = filtered_clf
    
    dists = clf.kneighbors(losses_np[available_ex], n_neighbors=n_neighbors)[0]
    if use_exp_weighting:
        exp_weighted_dists = np.log(oc_eps + dists.mean(axis=1))
        zero_min_dists = exp_weighted_dists - exp_weighted_dists.min()
        probs = zero_min_dists / zero_min_dists.max()
    else:
        probs = (dists - dists.min()) / (dists - dists.min()).max()  # Min 0, max 1 -- extremely naive.

    avail_ex_probs = np.array([probs, 1-probs]).T
    prob_values = np.linspace(0, 1, len(avail_ex_probs[:, 0]))
    final_probs = np.zeros((len(avail_ex_probs[:, 0])))
    final_probs[avail_ex_probs[:, 0].argsort()] = prob_values
    avail_ex_probs = np.array([final_probs, 1-final_probs]).T

    x, y, _ = roc_curve(np.array([1 if 'val' in x else 0 for x in dataset_probe_identity])[available_ex], avail_ex_probs[:, 0])
    print("AUC:", auc(x, y))
    record_detection(auc=auc(x, y))
    # TODO: Get rid of second row?
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
    thresh_list = [0.6]  # if attack == "all" else [0.15, 0.3]  # [0.25] if dataset == "imagenet" else [0.1, 0.25, 0.5, 0.75, 0.9]
    print("Threshold list:", thresh_list)

    output_checkpoint_dir = os.path.join(experiment_output_dir, "model_ft")
    if not os.path.exists(output_checkpoint_dir):
        os.makedirs(output_checkpoint_dir)
        print("!! Checkpoint output directory created:", output_checkpoint_dir)

    # train_types = ["original", "cleaned", "random"]
    train_types = ["cleaned"]
    # breakpoint()
    for train_type in train_types:
        print("=" * 100)
        print(f"!! Using {train_type} training set....")

        current_thresh_list = [None] if train_type == "original" else thresh_list
        for threshold in current_thresh_list:
            wandb_prefix = f"{train_type}_thresh_{threshold}_"
            if train_type == "original":
                # Use the training set w/o attacks
                assert threshold is None, threshold
                new_train_set_dl = get_loader(IdxDataset(train_set_wo_aug), seed=seed, distributed=distributed,
                                              num_workers=num_workers, batch_size=batch_size)
                title = "Retraining on the original train set (w/o backdoors)"
            else:
                assert threshold is not None, threshold
                is_clean = all_ex_probs[:, 0] <= threshold  # probability of an example being the backdoor is less than thresh
                clean_indices = np.where(is_clean)[0]

                if train_type == "cleaned":  # Remove examples marked as backdoors
                    print(f"!! [Dataset cleansing] Total examples: {len(is_clean)} / # clean indices: {len(clean_indices)}")
                    discarded_indices = [i for i in range(len(all_ex_probs)) if i not in clean_indices and
                                         i not in missing_vals_idx]
                    probe_identity_discarded_samples = [dataset_probe_identity[i] for i in discarded_indices]
                    print("!! Discarded example identities:", Counter(probe_identity_discarded_samples))
                    record_detection(num_removed=len(discarded_indices),
                                     removed_identities=dict(Counter(probe_identity_discarded_samples)))
                    
                    new_train_set_dl = get_loader(idx_dataset, seed=seed, indices=clean_indices, distributed=distributed,
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
                new_train_set_dl = get_loader(idx_dataset, seed=seed, indices=selected_indices, distributed=distributed,
                                              num_workers=num_workers, batch_size=batch_size)

            clean_model = get_model(args.dataset, num_classes, device, local_rank, verbose=False, arch=args.arch)

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
                                             attacks, tensor_batch_size)

            add_clean_to_output_dict(output_dict, clean_model, device, criterion, test_idx_loader, distributed, rank,
                                     log_predictions)
            results["retrained"] = {"clean_acc": output_dict["clean"]["accuracy"],
                                    "asr": {k: v["accuracy"] for k, v in output_dict.items() if k != "clean"}}
            save_results()
            output_file = os.path.join(experiment_output_dir, f"attack_success_{train_type}{postfix}.png")
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
                        verbose=True, per_class_out=None):
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

    if per_class_out is not None:
        per_class_out.update(per_class)
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

    retrain_set_dl = get_loader(idx_dataset, seed=seed, distributed=distributed, num_workers=num_workers,
                                  indices=retrain_indices, batch_size=batch_size)

    clean_model = get_model(args.dataset, num_classes, device, local_rank, verbose=False, arch=args.arch)
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
    retrained_test_stats = log_results_and_update_stats_and_preds(
        log_predictions, clean_model, device, clean_criterion, test_idx_loader, distributed, rank, retrain_set_dl,
        attack_types, probes, val_probes, defense, tensor_batch_size, num_epochs+1)
    retrained_asr = test_unseen_probes(log_predictions, clean_model, device, clean_criterion, test_probes, attacks,
                                       tensor_batch_size)
    record_model_metrics("retrained", retrained_test_stats, retrained_asr)
    return clean_model


wandb_prefix = "retraining_"
if args.defense == "nc":
    def apply_mask_and_trigger(batch, mask, trigger):
        return batch * (1 - mask) + mask * trigger


    def train_cleanse(model, mask, trigger, optimizer, target_class, l1_penalty, train_set,
                      use_autocast=False, log_interval=5):
        optimizer.zero_grad()
        pbar = tqdm(new_idx_loader_wo_aug)
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

    cleanse_epochs = args.nc_cleanse_epochs

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

    clean_activations, clean_indices, _, _ = get_last_layer_activations(model, new_idx_loader_wo_aug)
    clean_probe_indices = np.array(train_probe['clean_idx'])
    indices_to_check = torch.isin(clean_indices, torch.tensor(clean_probe_indices)).nonzero()[:, 0]

    poison_acts_by_class = []

    for atk_class in attacked_classes:
        # Get all activations
        mask, trigger = masks[atk_class], triggers[atk_class]
        dirty_activations, dirty_indices, _, dirty_predictions = get_last_layer_activations(model, new_idx_loader_wo_aug,
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
    det_auc = get_auc(auc_idx, valid_idx, dataset_probe_identity, args.num_train_probes)
    print("NC AUC", det_auc)
    record_detection(auc=det_auc)

    # Retrain model using default threshold
    identified_indices = get_indices_for_thresh(fpr_thresh, clean_probe_indices, attacked_classes,
                                                poison_acts_by_class, indices_to_check, clean_indices)

    # Print confusion stats...
    per_class_det = {}
    det_fpr, det_tpr = get_confusion_stats(identified_indices, valid_idx, dataset_probe_identity,
                                           args.num_train_probes, per_class_out=per_class_det)
    record_detection(fpr=det_fpr, tpr=det_tpr, num_removed=len(identified_indices), detection_rate=per_class_det)
    clean_model = retrain_model(identified_indices, comb_train_indices, idx_dataset, distributed, num_workers,
                                batch_size, num_classes, device, local_rank, lr, momentum, wd, num_epochs,
                                experiment_output_dir, fpr_thresh, probes, log_predictions, test_probes, args.defense)

    # fpr_thresh is only misnamed parameter...
    print("Done with nc")
    # TODO: retrain, repeat...
    # This doesn't work because the norms of the selected classes are higher than the median! Can rule these out,
    #   but that would mean that, by default, NC would pick up on nothing!!
    # Solution... somehow check the difference between high and lows?? This is a substantial extension...


if args.defense == "ac":
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

    activations, indices, classes, predictions = get_last_layer_activations(model, new_idx_loader_wo_aug)
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

    det_auc = get_auc(auc_idx, valid_idx, dataset_probe_identity, args.num_train_probes)
    print("AC AUC", det_auc)
    record_detection(auc=det_auc)

    # Retrain model using default threshold
    identified_indices = get_indices_for_class_clusters(detected_classes, clusterings, activation_by_predicted_class)
    per_class_det = {}
    det_fpr, det_tpr = get_confusion_stats(identified_indices, valid_idx, dataset_probe_identity,
                                           args.num_train_probes, per_class_out=per_class_det)
    record_detection(fpr=det_fpr, tpr=det_tpr, num_removed=len(identified_indices), detection_rate=per_class_det)
    clean_model = retrain_model(identified_indices, comb_train_indices, idx_dataset, distributed, num_workers,
                                batch_size, num_classes, device, local_rank, lr, momentum, wd, num_epochs,
                                experiment_output_dir, detect_thresh, probes, log_predictions, test_probes,
                                args.defense)  # detect_thresh...

    # Use probes to get expected clean silhouette scores per class
    # Get silhouette score -- if it's far from clean, then mark the smaller cluster as dirty. Repeat through all classes.
    # Ex-Re? It seems too difficult to do Ex-Re 20x training runs for each of 10 classes,
    #   then do AC all over again on the retrained model to find the second backdoor...

if args.defense == "ss":
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

    activations, indices, classes, predictions = get_last_layer_activations(model, new_idx_loader_wo_aug)
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

    det_auc = get_auc(auc_idx, valid_idx, dataset_probe_identity, args.num_train_probes)
    print("SS AUC", det_auc)
    record_detection(auc=det_auc)

    # Retrain model using default threshold
    identified_indices = get_indices_for_eps(epsilon_thresh, num_classes, taus, cls_idx, indices)

    # Print confusion stats...
    per_class_det = {}
    det_fpr, det_tpr = get_confusion_stats(identified_indices, valid_idx, dataset_probe_identity,
                                           args.num_train_probes, per_class_out=per_class_det)
    record_detection(fpr=det_fpr, tpr=det_tpr, num_removed=len(identified_indices), detection_rate=per_class_det)
    clean_model = retrain_model(identified_indices, comb_train_indices, idx_dataset, distributed, num_workers,
                                batch_size, num_classes, device, local_rank, lr, momentum, wd, num_epochs,
                                experiment_output_dir, epsilon_thresh, probes, log_predictions, test_probes, args.defense)

    print("Done with ss")
    # TODO: retrain, repeat...


if args.defense == "freq":
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
        'backdoor': copy.deepcopy(train_probe['clean'].cpu().numpy()),
    }

    def apply_random_transform(probe):
        attack = np.random.randint(0, 5)
        patch_x = np.random.randint(2, 8)
        patch_y = np.random.randint(2, 8)
        loc = np.random.randint(0, 6)
        corner = np.random.randint(0, 4)

        attack = np.random.randint(0, 2)

        if attack < 2:
            if attack == 0:
                block = np.ones((3, patch_x, patch_y))
            elif attack == 1:
                block = np.random.rand(3, patch_x, patch_y)

            if corner == 0:
                probe[:, loc:loc+patch_x, loc:loc+patch_y] = block
            elif corner == 1:
                probe[:, loc:loc+patch_x, -(loc+patch_y):-loc or None] = block
            elif corner == 2:
                probe[:, -(loc+patch_x):-loc or None, loc:loc+patch_y] = block
            elif corner == 3:
                probe[:, -(loc+patch_x):-loc or None, -(loc+patch_y):-loc or None] = block

        elif attack == 2:
            mean = 25
            var = np.random.uniform(10, 70)
            noise = np.random.randn(*probe.shape) * var + mean
            probe += noise / 255
        elif attack == 3:
            # very simplified version, thanks claude -- TODO?
            h, w = probe.shape[1:]
            top_y = np.random.randint(0, h)
            bottom_y = np.random.randint(top_y + 1, h + 1)
            left_x = np.random.randint(0, w)
            right_x = np.random.randint(left_x + 1, w + 1)

            shadow_mask = np.ones(probe.shape)
            shadow_mask[top_y:bottom_y, left_x:right_x] = 0.5  # 50% darkness

            # Apply shadow
            probe = probe * shadow_mask
        elif attack == 4:
            randind = np.random.randint(freq_probes['clean'].shape[0])
            blend_im = freq_probes['clean'][randind]
            probe = probe + 0.3 * blend_im

        return np.clip(probe, 0, 1)

    for i in range(freq_probes['backdoor'].shape[0]):
        probe = freq_probes['backdoor'][i]
        new_probe = apply_random_transform(probe)
        freq_probes['backdoor'][i] = new_probe


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
    freq_dataloader = DataLoader(freq_dataset, batch_size=32, shuffle=True,
                                 generator=torch.Generator().manual_seed(seed))

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

    det_auc = get_auc(auc_idx, valid_idx, dataset_probe_identity, args.num_train_probes)
    print("Freq AUC", det_auc)
    record_detection(auc=det_auc)

    # Retrain model using default threshold
    identified_indices = get_indices_for_thresh_from_loader(detection_thresh, new_idx_loader_wo_aug, freq_model)
    per_class_det = {}
    det_fpr, det_tpr = get_confusion_stats(identified_indices, valid_idx, dataset_probe_identity,
                                           args.num_train_probes, per_class_out=per_class_det)
    record_detection(fpr=det_fpr, tpr=det_tpr, num_removed=len(identified_indices), detection_rate=per_class_det)
    retrain_model(identified_indices, comb_train_indices, idx_dataset, distributed, num_workers, batch_size,
                  num_classes, device, local_rank, lr, momentum, wd, num_epochs, experiment_output_dir,
                  detection_thresh, probes, log_predictions, test_probes, args.defense)

if args.defense == "abl":
    def get_indices_from_losses(thresh, train_set, loss_idx, ex_idx):
        num_ex_unlearning = int(len(train_set) * thresh)
        identified_indices = loss_idx[:num_ex_unlearning]
        identified_indices = [int(ex_idx[i]) for i in identified_indices]
        return np.array(identified_indices)

    num_pretrain_epochs = args.abl_pretrain_epochs
    flooding_threshold = 0.5

    model = get_model(args.dataset, num_classes, device, local_rank, verbose=False, arch=args.arch)

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
                                                       val_probes, args.defense, tensor_batch_size, epoch)
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
    det_auc = get_auc(auc_idx, valid_idx, dataset_probe_identity, args.num_train_probes)
    print("ABL AUC", det_auc)
    record_detection(auc=det_auc)

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
        per_class_det = {}
        det_fpr, det_tpr = get_confusion_stats(identified_indices, valid_idx, dataset_probe_identity,
                                               args.num_train_probes, per_class_out=per_class_det)
        record_detection(fpr=det_fpr, tpr=det_tpr, num_removed=len(identified_indices), detection_rate=per_class_det)
        retrain_model(identified_indices, comb_train_indices, idx_dataset, distributed, num_workers, batch_size,
                      num_classes, device, local_rank, lr, momentum, wd, num_epochs, experiment_output_dir,
                      selection_threshold, probes, log_predictions, test_probes, args.defense)

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
        clean_dl = get_loader(new_idx_loader.dataset, seed=seed, distributed=distributed, indices=remaining_indices,
                              num_workers=num_workers, batch_size=batch_size)
        detected_backdoors_dl = get_loader(new_idx_loader.dataset, seed=seed, distributed=distributed, indices=indices_to_maximize,
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
                                                           val_probes, args.defense, tensor_batch_size, epoch)
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
                                                           val_probes, args.defense, tensor_batch_size, epoch)
            torch.save(model.state_dict(), output_checkpoint_file)
        else:
            print(f"!! Loading unlearned checkpoint file:", output_checkpoint_file)
            model.load_state_dict(torch.load(output_checkpoint_file, map_location=device))

        test_stats, test_preds = test(model, device, criterion, test_idx_loader, distributed, rank,
                                      log_predictions=log_predictions)
        test_unseen_probes(log_predictions, model, device, criterion, test_probes, attacks, tensor_batch_size)

if args.defense == "cd":
    cd = CognitiveDistillation(num_steps=args.cd_num_steps)
    masks = torch.zeros(len(new_idx_loader_wo_aug.dataset), *img_size[:-1])
    pbar = tqdm(new_idx_loader_wo_aug)
    for batch_idx, ((data, target), ex_idx) in enumerate(pbar):
        masks[ex_idx] = cd(model, data.to(device)).squeeze()

    mask_norms = torch.norm(masks, dim=(1, 2), p=1)
    valid_mask_idx = torch.where(mask_norms != 0)[0]
    clean_idx = train_probe['clean_idx']
    base_idx = np.intersect1d(clean_idx, valid_mask_idx.numpy())
    print("Num training examples for cognitive distillation", len(base_idx))

    # Not actually necessary, but good if we want to do their thresholded detection
    #   (everything less than -1 or -0.5 marked suspicious)
    mean, std = mask_norms[base_idx].mean(), mask_norms[base_idx].std()
    mask_norms[valid_mask_idx] -= mean
    mask_norms[valid_mask_idx] /= std

    # Like usual, remove bottom 15% and retrain?
    sorted_idx = mask_norms.argsort()
    detection_thresh = 0.15
    identified_indices = sorted_idx[:int(detection_thresh*len(mask_norms))]

    per_class_det = {}
    det_fpr, det_tpr = get_confusion_stats(identified_indices, valid_idx, dataset_probe_identity,
                                           args.num_train_probes, per_class_out=per_class_det)
    record_detection(fpr=det_fpr, tpr=det_tpr, num_removed=len(identified_indices), detection_rate=per_class_det)
    retrain_model(identified_indices, comb_train_indices, idx_dataset, distributed, num_workers, batch_size,
                  num_classes, device, local_rank, lr, momentum, wd, num_epochs, experiment_output_dir,
                  detection_thresh, probes, log_predictions, test_probes, args.defense)

if args.defense == "cbd":
    num_pretrain_epochs = args.cbd_pretrain_epochs

    backdoor_model = get_model(args.dataset, num_classes, device, local_rank, verbose=False, arch=args.arch)

    # Step # 01: Regular pretraining
    print("!! Performing CBD initial pretraining...")
    output_checkpoint_file = os.path.join(experiment_output_dir, "cbd_model_pretrain.pth")
    if not os.path.exists(output_checkpoint_file):
        criterion, optimizer, lr_scheduler, scaler = get_optimizer(backdoor_model, device, lr, momentum, wd, num_pretrain_epochs)
        for epoch in tqdm(range(num_pretrain_epochs)):
            train(backdoor_model, device, new_idx_loader, optimizer, criterion, scaler)
            if epoch % 5 == 4:
                log_results_and_update_stats_and_preds(log_predictions, backdoor_model, device, criterion, test_idx_loader,
                                                       distributed, rank, new_idx_loader_wo_aug, attack_types, probes,
                                                       val_probes, args.defense, tensor_batch_size, epoch)
        torch.save(backdoor_model.state_dict(), output_checkpoint_file)
    else:
        print(f"!! Loading pretrained checkpoint file:", output_checkpoint_file)
        backdoor_model.load_state_dict(torch.load(output_checkpoint_file, map_location=device))
    
    clean_model = get_model(args.dataset, num_classes, device, local_rank, verbose=False, arch=args.arch)
    
    discriminator = DisenEstimator(2048, 2048, dropout=0.2) # TODO: Magic number batd
    adv_params = list(discriminator.parameters())
    adv_optimizer = torch.optim.Adam(adv_params, lr=0.2)
    adv_scheduler = torch.optim.lr_scheduler.StepLR(adv_optimizer, step_size=20, gamma=0.1)
    optimizer = torch.optim.SGD(clean_model.parameters(), lr=0.1, momentum=0.9,
                                weight_decay=1e-4, nesterov=True)
    
    clean_train_epochs = num_epochs

    output_checkpoint_file = os.path.join(experiment_output_dir, "cbd_model.pth")
    if not os.path.exists(output_checkpoint_file):
        criterion = torch.nn.CrossEntropyLoss(reduction='none').to(device)
        scaler = None
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 70], gamma=0.1)
        for epoch in range(clean_train_epochs):
            train_cbd(clean_model, backdoor_model, discriminator, device, new_idx_loader, optimizer, adv_optimizer, criterion, args.cbd_ce_gamma)
            if epoch % 5 == 4:
                print(f"Evaluation at epoch {epoch+1}")
                log_results_and_update_stats_and_preds(log_predictions, clean_model, device, criterion, test_idx_loader,
                                                       distributed, rank, new_idx_loader_wo_aug, attack_types, probes,
                                                       val_probes, args.defense, tensor_batch_size, epoch)
            if epoch % 50 == 49:
                test_unseen_probes(log_predictions, clean_model, device, criterion, test_probes, attacks, tensor_batch_size)

            scheduler.step()
            adv_scheduler.step()
        torch.save(clean_model.state_dict(), output_checkpoint_file)
    else:
        print(f"!! Loading pretrained checkpoint file:", output_checkpoint_file)
        clean_model.load_state_dict(torch.load(output_checkpoint_file, map_location=device))

    print("Retrained model performance:")
    retrained_test_stats = log_results_and_update_stats_and_preds(
        log_predictions, clean_model, device, criterion, test_idx_loader, distributed, rank, new_idx_loader_wo_aug,
        attack_types, probes, val_probes, args.defense, tensor_batch_size, num_epochs+1)
    retrained_asr = test_unseen_probes(log_predictions, clean_model, device, criterion, test_probes, attacks,
                                       tensor_batch_size)
    record_model_metrics("retrained", retrained_test_stats, retrained_asr)

if args.defense == "pss":
    # Train from clean for 2 epochs w/o aug || train_attack_noTrans.py
    num_pretrain_epochs = args.pss_pretrain_epochs
    num_intraclass_epochs = args.pss_intraclass_epochs

    backdoor_model = get_model(args.dataset, num_classes, device, local_rank, verbose=False, arch=args.arch)
    optimizer = torch.optim.SGD(backdoor_model.parameters(), lr=0.01, momentum=0.9,
                                weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    scaler = None


    # Step # 01: Regular pretraining
    print("!! Performing PSS initial pretraining...")
    output_checkpoint_file = os.path.join(experiment_output_dir, "pss_model_pretrain.pth")
    if not os.path.exists(output_checkpoint_file):
        criterion = torch.nn.CrossEntropyLoss(reduction='none').to(device)
        scaler = None
        for epoch in tqdm(range(num_pretrain_epochs)):
            train(backdoor_model, device, new_idx_loader_wo_aug, optimizer, criterion, scaler)
            if epoch % 2 == 1:
                log_results_and_update_stats_and_preds(log_predictions, backdoor_model, device, criterion, test_idx_loader,
                                                       distributed, rank, new_idx_loader_wo_aug, attack_types, probes,
                                                       val_probes, args.defense, tensor_batch_size, epoch)
        torch.save(backdoor_model.state_dict(), output_checkpoint_file)
    else:
        print(f"!! Loading pretrained checkpoint file:", output_checkpoint_file)
        backdoor_model.load_state_dict(torch.load(output_checkpoint_file, map_location=device))
    

    # TODO!!!
    
    # Train intraclass loss for 10 epochs w/o aug || finetune_attack_noTrans.py
    optimizer = torch.optim.SGD(backdoor_model.parameters(), lr=0.01, momentum=0.9,
                                weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    train_criterion = torch.nn.CrossEntropyLoss().to(device)

    output_checkpoint_file = os.path.join(experiment_output_dir, "pss_model_intraclass.pth")
    print("!! Performing intraclass training...")
    if not os.path.exists(output_checkpoint_file):
        for epoch in tqdm(range(num_intraclass_epochs)):
            train_intraclass(backdoor_model, device, new_idx_loader_wo_aug, optimizer, train_criterion, scaler, num_classes)
            scheduler.step()
            if epoch % 5 == 4:
                log_results_and_update_stats_and_preds(log_predictions, backdoor_model, device, criterion, test_idx_loader,
                                                       distributed, rank, new_idx_loader_wo_aug, attack_types, probes,
                                                       val_probes, args.defense, tensor_batch_size, epoch)
        torch.save(backdoor_model.state_dict(), output_checkpoint_file)
    else:
        print(f"!! Loading intraclass trained checkpoint file:", output_checkpoint_file)
        backdoor_model.load_state_dict(torch.load(output_checkpoint_file, map_location=device))
    

    # Calculate FCT and sort samples || calculate_consistency.py & calculate_gamma.py & separate_samples.py
    
    high_thresh, low_thresh = 0.95, 0.80 # Inverted 
    print("!! Calculating FCT metric...")
    fcts = calc_fct(backdoor_model, device, new_idx_loader_wo_aug)
    fcts = fcts.cpu()

    classifications = [None for _ in range(len(fcts))]
    nonzero_fcts = fcts[fcts.nonzero()[:, 0]]
    sorted_fcts = nonzero_fcts.sort()[0]
    lower_limit = sorted_fcts[int(len(sorted_fcts) * low_thresh)].item()
    upper_limit = sorted_fcts[int(len(sorted_fcts) * high_thresh)].item()

    clean_idx = torch.where((fcts > 0) & (fcts < lower_limit))[0]
    pois_idx = torch.where(fcts >= upper_limit)[0]

    clean_dl = get_loader(new_idx_loader.dataset, seed=seed, distributed=distributed, indices=clean_idx,
                          num_workers=num_workers, batch_size=batch_size)
    pois_dl = get_loader(new_idx_loader.dataset, seed=seed, distributed=distributed, indices=pois_idx,
                         num_workers=num_workers, batch_size=batch_size)


    # Train backdoored model for 20 epochs of alternating learning and unlearning || unlearn_relearn.py    
    retrain_epochs = args.pss_unlearn_epochs
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9,
                                weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss(reduction='none').to(device)

    print("!! Performing PSS backdoor defense...")
    output_checkpoint_file = os.path.join(experiment_output_dir, "pss_model.pth")
    if not os.path.exists(output_checkpoint_file):
        for epoch in tqdm(range(retrain_epochs)):
            pss_unlearn(model, device, clean_dl, pois_dl, optimizer, criterion)
            if epoch % 5 == 4:
                log_results_and_update_stats_and_preds(log_predictions, model, device, criterion, test_idx_loader,
                                                       distributed, rank, new_idx_loader_wo_aug, attack_types, probes,
                                                       val_probes, args.defense, tensor_batch_size, epoch)
        torch.save(model.state_dict(), output_checkpoint_file)
    else:
        print(f"!! Loading PSS-trained checkpoint file:", output_checkpoint_file)
        model.load_state_dict(torch.load(output_checkpoint_file, map_location=device))

    print("Retrained model performance:")
    retrained_test_stats, _ = test(model, device, criterion, test_idx_loader, distributed, rank,
                                   log_predictions=log_predictions)
    retrained_asr = test_unseen_probes(log_predictions, model, device, criterion, test_probes, attacks,
                                       tensor_batch_size)
    record_model_metrics("retrained", retrained_test_stats, retrained_asr)

    print("Done with PSS")


if log_wandb:
    wandb.finish()
