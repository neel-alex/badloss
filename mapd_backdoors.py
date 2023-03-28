#!/usr/bin/env python
# coding: utf-8

# ## Inserting probes into the model for inspecting model phase

# In[ ]:


import os
import sys
import copy
import shutil
import pickle
import natsort
import warnings
import itertools
from tqdm import tqdm
from collections import OrderedDict

import random

import torch
from torchvision.datasets import MNIST, CIFAR10, CIFAR100, ImageFolder
from torchvision import transforms, models

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import cv2

import urllib

import dist_utils

from catalyst.data import DistributedSamplerWrapper

import sklearn.neighbors
from sklearn.metrics import confusion_matrix


# In[ ]:


# Set random seed
seed = 3
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed=seed)
random.seed(seed)


# In[ ]:


# Plotting config
include_plot_title = False
font_size = 16


# In[ ]:


if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} <Dataset: mnist/cifar10/cifar100/imagenet>")
    exit()

dataset = sys.argv[1]
loss_type = "ce"
assert dataset in ["mnist", "cifar10", "cifar100", "imagenet"]
assert loss_type in ["ce", "ce_supcon", "ce_center"]


# In[ ]:


# Essential config
log_predictions = True
distributed = True if dataset == "imagenet" else False
num_train_probes = 250
num_val_probes = 250
use_val_probes_for_training = True
num_example_probes = num_train_probes + num_val_probes
experiment_output_dir = f"./backdoor_exp04_{dataset}_{loss_type}_alpha_0.1"
num_workers = 8
surface_examples = False
aux_loss_lambda = 1.0  # Based on the experiments with center loss
feat_dim = 2048  # Feature dimensions for ResNet-50
dataset_name = dataset

print("Dataset:", dataset)
print("Loss function:", loss_type)
print("Distributed training:", distributed)


# In[ ]:


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


# In[ ]:


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


# In[ ]:


setup_for_distributed(main_proc)
warnings.filterwarnings("ignore", "Warning: Leaking Caffe2 thread-pool after fork. (function pthreadpool)", UserWarning)


# In[ ]:


recompute_results = False
if main_proc:
    if recompute_results:
        if os.path.exists(experiment_output_dir):
            shutil.rmtree(experiment_output_dir)
        os.makedirs(experiment_output_dir)
    else:
        if not os.path.exists(experiment_output_dir):
            os.makedirs(experiment_output_dir)


# In[ ]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Current device:", device)


# In[ ]:


def load_class_mapping(dataset):
    if dataset == "imagenet":
        # Load idx to class name mapping
        url = "https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt"
        response = urllib.request.urlopen(url)
        lines = response.readlines()
        string = ''.join([line.decode("utf-8") for line in lines])
        label2name = eval(string)
    elif dataset == "imagenette":
        classes = ["tench", "english springer", "cassette player", "chain saw", "church", "french horn", "garbage truck",
                   "gas pump", "golf ball", "parachute"]
        label2name = {k: v for k, v in enumerate(classes)}
    elif dataset == "cifar10":
        classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
        label2name = {k: v for k, v in enumerate(classes)}
    elif dataset == "cifar100":
        classes = ["beaver", "dolphin", "otter", "seal", "whale", "aquarium fish", "flatfish", "ray", "shark", "trout",
                   "orchids", "poppies", "roses", "sunflowers", "tulips", "bottles", "bowls", "cans", "cups", "plates",
                   "apples", "mushrooms", "oranges", "pears", "sweet peppers", "clock", "computer keyboard", "lamp",
                   "telephone", "television", "bed", "chair", "couch", "table", "wardrobe", "bee", "beetle", "butterfly",
                   "caterpillar", "cockroach", "bear", "leopard", "lion", "tiger", "wolf", "bridge", "castle", "house",
                   "road", "skyscraper", "cloud", "forest", "mountain", "plain", "sea", "camel", "cattle", "chimpanzee",
                   "elephant", "kangaroo", "fox", "porcupine", "possum", "raccoon", "skunk", "crab", "lobster", "snail",
                   "spider", "worm", "baby", "boy", "girl", "man", "woman", "crocodile", "dinosaur", "lizard", "snake",
                   "turtle", "hamster", "mouse", "rabbit", "shrew", "squirrel", "maple", "oak", "palm", "pine", "willow",
                   "bicycle", "bus", "motorcycle", "pickup truck", "train", "lawn-mower", "rocket", "streetcar", "tank", "tractor"]
        label2name = {k: v for k, v in enumerate(classes)}
    else:
        raise RuntimeError(f"Unknown dataset: {dataset}")
    name2label = {label2name[key]: key for key in label2name.keys()}
    assert name2label[label2name[0]] == 0
    return label2name, name2label


# In[ ]:


if "mnist" in dataset:
    img_size = (28, 28, 1)
    train_transform = [transforms.ToTensor()]
    test_transform = [transforms.ToTensor()]
    no_transform = test_transform
    
    data_dir = f"/mnt/sas/Datasets/{dataset}/"  # TODO: Configure dataset path
    train_set = MNIST(data_dir, download=True, train=True, transform=transforms.Compose(train_transform))
    test_set = MNIST(data_dir, download=True, train=False, transform=transforms.Compose(test_transform))
elif "cifar" in dataset:
    img_size = (32, 32, 3)
    train_transform = [transforms.RandomHorizontalFlip(),
                       transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
                       transforms.ToTensor()]
    test_transform = [transforms.ToTensor()]
    no_transform = test_transform
    
    DatasetCls = CIFAR100 if dataset == "cifar100" else CIFAR10 if dataset == "cifar10" else None
    assert DatasetCls is not None
    
    data_dir = f"/mnt/sas/Datasets/{dataset}/"  # TODO: Configure dataset path
    train_set = DatasetCls(data_dir, download=True, train=True, transform=transforms.Compose(train_transform))
    train_set_wo_aug = DatasetCls(data_dir, download=True, train=True, transform=transforms.Compose(no_transform))
    test_set = DatasetCls(data_dir, download=True, train=False, transform=transforms.Compose(test_transform))
else:
    assert dataset == "imagenet"
    img_size = (224, 224, 3)
    
    use_augmentations = True
    if use_augmentations:
        print("Training w/ augmentations...")
        train_transform = [transforms.RandomResizedCrop(224),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor()]
        test_transform = [transforms.Resize(256),
                          transforms.CenterCrop(224),
                          transforms.ToTensor()]
        no_transform = [transforms.Resize((224, 224)),
                        transforms.ToTensor()]
    else:
        print("Training w/o augmentations...")
        train_transform = [transforms.Resize((224, 224)),
                           transforms.ToTensor()]
        test_transform = [transforms.Resize((224, 224)),
                          transforms.ToTensor()]
    no_transform = test_transform

    data_dir = "/ds/images/imagenet/"  # TODO: Configure dataset path
    train_set = ImageFolder(os.path.join(data_dir, "train"), transform=transforms.Compose(train_transform))
    train_set_wo_aug = ImageFolder(os.path.join(data_dir, "train"), transform=transforms.Compose(no_transform))
    test_set = ImageFolder(os.path.join(data_dir, "val_folders"), transform=transforms.Compose(test_transform))
    
    # Replace train_set.classes with real names
    train_set.original_classes = train_set.classes
    
    label2name, _ = load_class_mapping(dataset)
    label2name = {k: v.split(',')[0][:20] for k, v in label2name.items()}
    train_set.classes = label2name  # Dict mapping from label to class name

print(dataset, len(train_set), len(test_set))


# In[ ]:


def get_loader(dataset, indices=None, batch_size=16, shuffle=False):
    sampler = None
    if indices is not None:
        sampler = torch.utils.data.SubsetRandomSampler(indices)
    if distributed:
        if sampler is not None:
            print("Using distributed sampler on top of previous sampler...")
            sampler = DistributedSamplerWrapper(sampler)
        else:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler, num_workers=num_workers, pin_memory=True)
    return loader


# In[ ]:


def plot(x, y=None, memorization_val=None, class_names=None, output_file=None, add_mem_scores=False):
    num_plots_per_row = 3
    plot_rows = 3
    plot_size = 3
    fig, ax = plt.subplots(plot_rows, num_plots_per_row, figsize=(plot_size * num_plots_per_row, plot_size * plot_rows), sharex=True, sharey=True)

    input = x.cpu().numpy()
    is_grayscale = input.shape[1] == 1
    input = input[:, 0, :, :] if is_grayscale else np.transpose(input, (0, 2, 3, 1))
    if class_names is not None:
        y = [class_names[int(i)] for i in y]
    
    for idx in range(len(input)):
        ax[idx // num_plots_per_row, idx % num_plots_per_row].imshow(input[idx], cmap='gray' if is_grayscale else None)
        if y is not None:
            if add_mem_scores:
                ax[idx // num_plots_per_row, idx % num_plots_per_row].set_title(f"{'Label: ' if class_names is None else ''}{y[idx]}" + (f"\n(Mem: {memorization_val[idx]:.4f})" if memorization_val is not None else ""), color='g')
            else:
                ax[idx // num_plots_per_row, idx % num_plots_per_row].set_title(f"{'Label: ' if class_names is None else ''}{y[idx].replace('_', ' ').title()}", color='k', fontsize=font_size)

        if idx == plot_rows * num_plots_per_row - 1:
            break

    for a in ax.ravel():
        a.set_axis_off()
        a.set_yticklabels([])
        a.set_xticklabels([])

    fig.tight_layout()
    if output_file is not None:
        plt.savefig(os.path.join(experiment_output_dir, output_file), dpi=300, bbox_inches="tight")
    plt.show()
    plt.close('all')


# ## Setup probes

# In[ ]:


num_classes = len(train_set.classes)
assert num_classes == 10 if dataset in ["mnist", "cifar10"] else num_classes == 100 if dataset == "cifar100" else num_classes == 1000
print(dataset, num_classes)


# In[ ]:


class BackdoorPatch(object):
    def __init__(self, single_pixel_backdoor=False, reverse_backdoor=False, 
                 pattern=None, alpha=None):
        assert pattern is None or (not single_pixel_backdoor and not reverse_backdoor and alpha is not None)
        
        self.single_pixel_backdoor = single_pixel_backdoor
        self.reverse_backdoor = reverse_backdoor
        
        self.pattern = pattern
        self.alpha = alpha
    
    def __call__(self, tensor):
        backdoor_pix_val = 1.0
        
        if self.pattern is not None:
            tensor = (1-self.alpha) * tensor + (self.alpha) * self.pattern
        elif self.single_pixel_backdoor:
            if self.reverse_backdoor:
                tensor[:, 1, 1] = backdoor_pix_val
            else:
                tensor[:, tensor.shape[1]-2, tensor.shape[2]-2] = backdoor_pix_val
        else:
            if self.reverse_backdoor:
                tensor[:, 1, 1] = backdoor_pix_val
                tensor[:, 3, 1] = backdoor_pix_val
                tensor[:, 1, 3] = backdoor_pix_val
                tensor[:, 2, 2] = backdoor_pix_val
            else:
                tensor[:, tensor.shape[1]-2, tensor.shape[2]-2] = backdoor_pix_val
                tensor[:, tensor.shape[1]-4, tensor.shape[2]-2] = backdoor_pix_val
                tensor[:, tensor.shape[1]-2, tensor.shape[2]-4] = backdoor_pix_val
                tensor[:, tensor.shape[1]-3, tensor.shape[2]-3] = backdoor_pix_val
        return tensor


# In[ ]:


class ClampRangeTransform(object):
    def __init__(self):
        pass
    
    def __call__(self, x):
        return torch.clamp(x, 0., 1.)


# In[ ]:


probes = {"backdoor": [], "clean": []}


# In[ ]:


fake_backdoor_label = np.random.choice(np.arange(num_classes))
print(f"!! Random class picked for original backdoor: {fake_backdoor_label}")
backdoor_transform = transforms.Compose([BackdoorPatch(), ClampRangeTransform()])

backdoor_idx = np.random.choice(np.arange(len(train_set)), size=num_example_probes, replace=False)
probes.update({"backdoor_idx": backdoor_idx[:num_example_probes]})
probes["backdoor"] = torch.stack([backdoor_transform(train_set[i][0]) for i in probes["backdoor_idx"]], dim=0).to(device)
probes["backdoor_labels"] = torch.from_numpy(np.array([fake_backdoor_label for i in probes["backdoor_idx"]])).to(device)
print("Backdoor probe shape:", probes["backdoor"].shape)


# In[ ]:


attack_types = ["reversed", "single_pix", "reversed_single_pix", "random"]
remaining_indices = [i for i in range(len(train_set)) if i not in probes["backdoor_idx"]]
new_indices = np.random.choice(remaining_indices, size=num_example_probes*len(attack_types), replace=False)
probes.update({"novel_backdoor_idx": new_indices})

# Create a main random pattern
pattern_file = os.path.join(experiment_output_dir, "random_pattern.png")
if main_proc:
    random_pattern = np.clip(np.random.rand(*img_size) * 255, 0, 255)
    print("Random pattern shape:", random_pattern.shape)
    cv2.imwrite(pattern_file, random_pattern)
dist_utils.wait_for_other_procs()  # Distributed barrier

# Load the pattern to ensure the same pattern is loaded by all processes
random_pattern_img = cv2.imread(pattern_file, cv2.IMREAD_UNCHANGED)
random_pattern = transforms.ToTensor()(random_pattern_img)
print(f"Random Pattern / Loaded shape: {random_pattern_img.shape} / Tensor shape: {random_pattern.shape} / Min: {random_pattern.min()} / Max: {random_pattern.max()}")

for i, attack_type in enumerate(attack_types):
    if attack_type == "random":
        backdoor = BackdoorPatch(pattern=random_pattern, alpha=0.1)
    elif attack_type == "reversed":
        backdoor = BackdoorPatch(reverse_backdoor=True)
    elif attack_type == "single_pix":
        backdoor = BackdoorPatch(single_pixel_backdoor=True)
    else:
        assert attack_type == "reversed_single_pix"
        backdoor = BackdoorPatch(single_pixel_backdoor=True, reverse_backdoor=True)
    backdoor_transform = transforms.Compose([backdoor, ClampRangeTransform()])
    
    fake_backdoor_label = np.random.choice(np.arange(num_classes))
    print(f"!! Random class picked for {attack_type.replace('_', ' ')} backdoor: {fake_backdoor_label}")
    
    current_idx = probes["novel_backdoor_idx"][i*num_example_probes:(i+1)*num_example_probes]
    probes[f"backdoor_{attack_type}"] = torch.stack([backdoor_transform(train_set[i][0]) for i in current_idx], dim=0).to(device)
    probes[f"backdoor_{attack_type}_labels"] = torch.from_numpy(np.array([fake_backdoor_label for i in current_idx])).to(device)
    print(f"Backdoor ({attack_type}) probe shape:", probes["backdoor"].shape)


# In[ ]:


remaining_indices = [i for i in range(len(train_set)) if i not in probes["backdoor_idx"] and i not in probes["novel_backdoor_idx"]]
new_indices = np.random.choice(remaining_indices, size=num_example_probes, replace=False)
probes.update({"clean_idx": new_indices})

probes["clean"] = torch.stack([train_set[i][0] for i in probes["clean_idx"]], dim=0).to(device)
probes["clean_labels"] = torch.from_numpy(np.array([train_set[i][1] for i in probes["clean_idx"]])).to(device)
print("Clean probe shape:", probes["clean"].shape)


# In[ ]:


print("Backdoor examples")
plot(probes["backdoor"], probes["backdoor_labels"], None, class_names=train_set.classes, output_file=f"backdoor_{dataset}_{rank}.png")


# In[ ]:


# Plot updated backdoors
print("Updated backdoor examples")
for attack_type in attack_types:
    plot(probes[f"backdoor_{attack_type}"], probes[f"backdoor_{attack_type}_labels"], None, class_names=train_set.classes, output_file=f"backdoor_{dataset}_{attack_type}_{rank}.png")


# In[ ]:


print("Clean examples")
plot(probes["clean"], probes["clean_labels"], None, class_names=train_set.classes, output_file=f"clean_{dataset}_{rank}.png")


# In[ ]:


# Hyperparameters
if dataset == "mnist":
    num_epochs = 25
    batch_size = 256
elif "cifar" in dataset:
    num_epochs = 150
    batch_size = 128
else:
    assert dataset == "imagenet"
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


# In[ ]:


discarded_idx = list(probes["backdoor_idx"]) + list(probes["novel_backdoor_idx"]) + list(probes["clean_idx"])
train_indices = [i for i in range(len(train_set)) if i not in discarded_idx]
print("Discarded examples:", len(train_set) - len(train_indices))
assert len(train_set) - len(train_indices) == len(discarded_idx)
train_loader = get_loader(train_set, train_indices, batch_size=batch_size)
test_loader = get_loader(test_set, batch_size=batch_size)


# In[ ]:


class CustomTensorDataset(torch.utils.data.Dataset):
    def __init__(self, x: torch.Tensor, y: list) -> None:
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.x.size(0)


# In[ ]:


probes_to_be_used = ["backdoor", "clean"]
print("Selected probes to be used:", probes_to_be_used)
val_idx = np.random.choice(range(num_example_probes), size=num_val_probes, replace=False)

# Filter the train indexes
val_probes = {}
probe_identity = []
val_probe_identity = []
for primary_k in probes_to_be_used:
    for suffix in ["", "_labels"]:
        k = f"{primary_k}{suffix}"
        print("Current key:", k)
        assert len(probes[k]) == num_example_probes
        shape_len = len(probes[k].shape)
        
        val_probes[k] = torch.cat([probes[k][i:i+1] for i in range(len(probes[k])) if i in val_idx], dim=0)  # Transfer val indices from train
        probes[k] = torch.cat([probes[k][i:i+1] for i in range(len(probes[k])) if i not in val_idx], dim=0)  # Discard val index from train
        
        assert len(val_probes[k].shape) == shape_len
        assert len(probes[k].shape) == shape_len

        assert len(val_probes[k]) == num_val_probes
        assert len(probes[k]) == num_train_probes
    
    probe_identity += [primary_k for _ in range(len(probes[primary_k]))]
    val_probe_identity += [f"{primary_k}_val" for _ in range(len(val_probes[primary_k]))]

# Add additional probes here
probes_to_be_used_val = [x for x in probes_to_be_used]  # Deep copy
for attack_type in attack_types:
    key = f"backdoor_{attack_type}"
    val_probes[key] = probes[key]
    val_probes[f"{key}_labels"] = probes[f"{key}_labels"]
    val_probe_identity += [f"{key}_val" for _ in range(len(val_probes[key]))]
    probes_to_be_used_val += [key]

probe_images = torch.cat([probes[k] for k in probes_to_be_used], dim=0)
probe_labels = torch.cat([probes[f"{k}_labels"] for k in probes_to_be_used], dim=0)
assert len(probe_identity) == len(probe_images), f"{len(probe_identity)} != {len(probe_images)}"

# Shuffle
perm = np.random.choice(range(len(probe_images)), size=len(probe_images), replace=False)
probe_images = torch.stack([probe_images[i] for i in perm], dim=0).to(device)
probe_labels = torch.stack([probe_labels[i] for i in perm], dim=0).to(device)
probe_identity = [probe_identity[i] for i in perm]
print(f"Probe | Images: {probe_images.shape} | Labels: {probe_labels.shape}")

probe_dataset = torch.utils.data.TensorDataset(probe_images, probe_labels)
probe_dataset_standard = CustomTensorDataset(probe_images.to("cpu"), [int(x) for x in probe_labels.to("cpu").numpy().tolist()])
print("Probe dataset:", len(probe_dataset_standard), probe_dataset_standard[0][0].shape, probe_dataset_standard[0][1])

# Create the validation set for probes
val_probe_images = torch.cat([val_probes[k] for k in probes_to_be_used_val], dim=0)
val_probe_labels = torch.cat([val_probes[f"{k}_labels"] for k in probes_to_be_used_val], dim=0)
val_probe_dataset_standard = CustomTensorDataset(val_probe_images.to("cpu"), [int(x) for x in val_probe_labels.to("cpu").numpy().tolist()])
print("Validation probe dataset:", len(val_probe_dataset_standard), val_probe_dataset_standard[0][0].shape, val_probe_dataset_standard[0][1])


# In[ ]:


class ProbeDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, probe_identity, remove_val_in_name=True):
        self.dataset = dataset
        if remove_val_in_name:
            probe_identity = [x.replace("_val", "") for x in probe_identity]
            print("Validation tag in probe identity removed for probe dataset...")
        self.probe_identity = probe_identity
        self.class_names = natsort.natsorted(np.unique(probe_identity))
        self.iden2label = {iden: i for i, iden in enumerate(self.class_names)}
        print("Probe to idx map:", self.iden2label)
    
    def get_probe_map(self):
        return self.iden2label
    
    def get_class_names(self):
        return self.class_names
    
    def __getitem__(self, idx):
        return self.dataset[idx], self.iden2label[self.probe_identity[idx]]


# In[ ]:


# Setup the probe validation set
val_probe_dataset = ProbeDataset(val_probe_dataset_standard, val_probe_identity)
val_probe_indices = [i for i in range(len(val_probe_dataset_standard))]
val_probe_loader = get_loader(val_probe_dataset, val_probe_indices, batch_size=batch_size)


# In[ ]:


print("Curated probe dataset")
plot(torch.stack([x[0] for x in probe_dataset], dim=0), torch.stack([x[1] for x in probe_dataset], dim=0), class_names=train_set.classes, output_file=f"probes_dataset_{dataset}.png")


# In[ ]:


if dataset_name == "mnist":
    # Create BadNet architecture (https://arxiv.org/abs/1708.06733)
    model = torch.nn.Sequential(OrderedDict([
        ('conv1', torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0)),
        ('act1', torch.nn.ReLU(inplace=True)),
        ('pool1', torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)),
        ('conv2', torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0)),
        ('act2', torch.nn.ReLU(inplace=True)),
        ('pool2', torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)),
        ('flatten', torch.nn.Flatten()),
        ('fc1', torch.nn.Linear(in_features=32*4*4, out_features=512)),
        ('fc1_act', torch.nn.ReLU(inplace=True)),
        ('fc', torch.nn.Linear(in_features=512, out_features=10)),
    ]))
    model = model.to(device)
else:
    # Create ResNet-50
    model = models.resnet50(pretrained=False)
    if "cifar" in dataset:  # Change the first and last layer for cifar10/cifar100
        model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)
print(model)


# In[ ]:


# Convert to a distributed model
model = dist_utils.convert_to_distributed(model, local_rank, sync_bn=True)


# In[ ]:


def train(model, device, train_loader, optimizer, criterion, scaler, log_interval=10, log_predictions=False, use_autocast=False):
    model.train()
    optimizer.zero_grad()
    
    example_idx = []
    predictions = []
    targets = []
    loss_values = []
    
    pbar = tqdm(train_loader)
    for batch_idx, ((data, target), ex_idx) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast(enabled=use_autocast):
            output = model(data)
            loss = criterion(output, target)
        
        loss_values.append(loss.detach().clone())
        
        assert loss.shape == (len(data),)
        loss = loss.mean()  # Reduction has been disabled -- do explicit reduction
        
        if use_autocast:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        if log_predictions:
            predictions.append(output.argmax(dim=1).detach())
            example_idx.append(ex_idx.clone())
            targets.append(target.clone())
        
        if batch_idx % log_interval == 0:
            pbar.set_description(f"Loss: {float(loss):.4f}")
        torch.cuda.synchronize()
    pbar.close()
    
    output_dict = None
    if log_predictions:
        # Collect the statistics from all the GPUs
        example_idx = torch.cat(dist_utils.gather_tensor(torch.cat(example_idx, dim=0)), dim=0).detach().cpu().numpy()
        predictions = torch.cat(dist_utils.gather_tensor(torch.cat(predictions, dim=0)), dim=0).detach().cpu().numpy()
        targets = torch.cat(dist_utils.gather_tensor(torch.cat(targets, dim=0)), dim=0).detach().cpu().numpy()
        loss_values = torch.cat(dist_utils.gather_tensor(torch.cat(loss_values, dim=0)), dim=0).detach().cpu().numpy()
        output_dict = {"ex_idx": example_idx, "preds": predictions, "targets": targets, "loss": loss_values}

    return output_dict


# In[ ]:


def test(model, device, criterion, test_loader, set_name="Test", log_predictions=False):
    model.eval()
    
    correct = torch.tensor([0]).to(device)
    test_loss = torch.tensor([0.0]).to(device)
    total = torch.tensor([0]).to(device)
    test_acc = 0.
    
    example_idx = []
    predictions = []
    targets = []
    loss_values = []
    
    for (data, target), ex_idx in test_loader:
        with torch.no_grad():
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss_vals = criterion(output, target)
            
            test_loss += float(loss_vals.sum())
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += len(data)
            
            if log_predictions:
                loss_values.append(loss_vals.detach().clone())
                predictions.append(output.argmax(dim=1).detach())
                example_idx.append(ex_idx.clone())
                targets.append(target.clone())
    
    # Reduce all of the values in case of distributed processing
    torch.cuda.synchronize()
    correct = int(dist_utils.reduce_tensor(correct.data))
    test_loss = float(dist_utils.reduce_tensor(test_loss.data))
    total = int(dist_utils.reduce_tensor(total.data))
    
    if isinstance(test_loader.sampler, torch.utils.data.distributed.DistributedSampler):
        num_dataset_ex = len(test_loader.sampler.dataset)
    elif isinstance(test_loader.sampler, DistributedSamplerWrapper):
        num_dataset_ex = len(test_loader.sampler.sampler.dataset)
    elif isinstance(test_loader.sampler, torch.utils.data.SubsetRandomSampler):
        num_dataset_ex = len(test_loader.sampler)
    else:
        # assert test_loader.sampler is None, test_loader.sampler
        num_dataset_ex = len(test_loader.dataset)
    
    if not distributed:
        assert total == num_dataset_ex, f"{total} != {num_dataset_ex}"
    if total != num_dataset_ex:
        print(f"!! Warning -- aggregated total value ({total}) is not equal to the dataset size: {num_dataset_ex}...")
    if total > 0:
        test_loss /= total
        test_acc = 100. * correct / total
    output_dict = dict(loss=test_loss, acc=test_acc, correct=correct, total=total)
    if distributed:
        set_name = f"Rank: {rank} | {set_name}"
    print(f"{set_name} set | Average loss: {test_loss:.4f} | Accuracy: {correct}/{total} ({test_acc:.2f}%)")
    
    pred_output_dict = None
    if log_predictions:
        # Collect the statistics from all the GPUs
        example_idx = torch.cat(dist_utils.gather_tensor(torch.cat(example_idx, dim=0)), dim=0).detach().cpu().numpy()
        predictions = torch.cat(dist_utils.gather_tensor(torch.cat(predictions, dim=0)), dim=0).detach().cpu().numpy()
        targets = torch.cat(dist_utils.gather_tensor(torch.cat(targets, dim=0)), dim=0).detach().cpu().numpy()
        loss_values = torch.cat(dist_utils.gather_tensor(torch.cat(loss_values, dim=0)), dim=0).detach().cpu().numpy()
        pred_output_dict = {"ex_idx": example_idx, "preds": predictions, "targets": targets, "loss": loss_values}
    return output_dict, pred_output_dict


# In[ ]:


def test_tensor(model, device, criterion, data, target, msg=None, log_predictions=False):
    assert torch.is_tensor(data) and torch.is_tensor(target)
    
    model.eval()
    with torch.no_grad():
        output = model(data)
        loss_vals = criterion(output, target)
        test_loss = float(loss_vals.mean())
        
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct = pred.eq(target.view_as(pred)).sum().item()
        total = len(data)
    
    test_acc = 100. * correct / total
    output_dict = dict(loss=test_loss, acc=test_acc, correct=correct, total=total)
    
    loss_vals = loss_vals.detach().cpu().numpy()
    output_dict["loss_mean"] = np.mean(loss_vals)
    output_dict["loss_var"] = np.var(loss_vals)
    output_dict["loss_std"] = np.std(loss_vals)
    
    pred_dict = None
    if log_predictions:
        pred_dict = {}
        pred_dict["ex_idx"] = np.arange(len(loss_vals))
        pred_dict["loss_vals"] = loss_vals
        pred_dict["preds"] = pred.detach().cpu().numpy()
        pred_dict["targets"] = target.detach().cpu().numpy()
    
    header = "Test set" if msg is None else msg
    print(f"{header} | Loss mean: {output_dict['loss_mean']:.4f} | Loss std: {output_dict['loss_std']:.4f} | Accuracy: {test_acc:.2f}% ({correct}/{total})")
    
    return output_dict, pred_dict


# In[ ]:


criterion = torch.nn.CrossEntropyLoss(reduction='none').to(device)  # reduction='mean' by default


# In[ ]:


optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
scaler = torch.cuda.amp.GradScaler()


# In[ ]:


# Combine the two datasets (probe dataset and normal dataset)
if use_val_probes_for_training:
    print("!! Including validation probes in the training process...")
    comb_train_set = torch.utils.data.ConcatDataset([train_set, probe_dataset_standard, val_probe_dataset_standard])
    comb_train_indices = train_indices + [(len(train_set) + x) for x in range(len(probe_dataset_standard)+len(val_probe_dataset_standard))]
else:
    comb_train_set = torch.utils.data.ConcatDataset([train_set, probe_dataset_standard])
    comb_train_indices = train_indices + [(len(train_set) + x) for x in range(len(probe_dataset_standard))]
print("Indices in combined dataset:", len(comb_train_indices))
assert len(np.unique(comb_train_indices)) == len(comb_train_indices)
print("Size of combined dataset:", len(comb_train_set))


# In[ ]:


dataset_probe_identity = ["train" for i in range(len(train_set))] + probe_identity
if use_val_probes_for_training:
    dataset_probe_identity += val_probe_identity
assert len(dataset_probe_identity) == len(comb_train_set), f"{len(dataset_probe_identity)} != {len(comb_train_set)}"


# In[ ]:


class IdxDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx], idx


# In[ ]:


# Convert into a dataset which returns indices
idx_dataset = IdxDataset(comb_train_set)
new_idx_loader = get_loader(idx_dataset, comb_train_indices, batch_size=batch_size)
train_idx_loader = get_loader(IdxDataset(train_set), train_indices, batch_size=batch_size)
test_idx_loader = get_loader(IdxDataset(test_set), batch_size=batch_size)


# In[ ]:


num_plots_per_row = 3
plot_rows = 3
num_queue_plots = num_plots_per_row * plot_rows


# In[ ]:


def plot_probe_ex(x, y, probs, output_file=None):
    plot_size = 3
    fig, ax = plt.subplots(plot_rows, num_plots_per_row, figsize=(plot_size * num_plots_per_row, plot_size * plot_rows), sharex=True, sharey=True)

    for idx in range(len(x)):
        ax[idx // num_plots_per_row, idx % num_plots_per_row].imshow(x[idx])
        # ax[idx // num_plots_per_row, idx % num_plots_per_row].set_title(y[idx])
        if probs is not None:
            ax[idx // num_plots_per_row, idx % num_plots_per_row].set_title(f"{y[idx]} (PD: {probs[idx]:.3f})")
        else:
            ax[idx // num_plots_per_row, idx % num_plots_per_row].set_title(f"{y[idx]}")

        if idx == plot_rows * num_plots_per_row - 1:
            break

    for a in ax.ravel():
        a.set_axis_off()

        # Turn off tick labels
        a.set_yticklabels([])
        a.set_xticklabels([])

    fig.tight_layout()
    if output_file is not None:
        fig.savefig(output_file, bbox_inches=0.0, pad_inches=0)
    plt.close()


# In[ ]:


model_file = os.path.join(experiment_output_dir, f"models_{dataset}", f"model_{dataset}_dynamics.pth")
data_file = os.path.join(experiment_output_dir, f"stats_{dataset}_dynamics.pkl")
data_statistics_file = os.path.join(experiment_output_dir, f"stats_{dataset}_data_statistics.pkl")
surface_dir = os.path.join(experiment_output_dir, f"./surfaced_examples_{dataset}/")
surface_clustering_dir = os.path.join(experiment_output_dir, f"./surfaced_clustering_examples_{dataset}/")
conf_mat_dir = os.path.join(experiment_output_dir, f"./confusion_matrix_{dataset}/")


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
    
    if main_proc:
        if os.path.exists(surface_dir):
            shutil.rmtree(surface_dir)
            print("Removed previous surfaced example output directory...")
        os.mkdir(surface_dir)
        
        if os.path.exists(conf_mat_dir):
            shutil.rmtree(conf_mat_dir)
            print("Removed previous confusion matrix directory...")
        os.mkdir(conf_mat_dir)
    
    for epoch in range(num_epochs):
        output_dict = train(model, device, new_idx_loader, optimizer, criterion, scaler)
        
        # Collect test set statistics
        print("Stats for epoch #", epoch+1)
        if log_predictions:
            # Don't use train_idx_loader here -- also assumes that probes are include for later evaluation
            train_stats, train_preds = test(model, device, criterion, new_idx_loader, set_name="Train", log_predictions=log_predictions)
        
        test_stats, test_preds = test(model, device, criterion, test_idx_loader, log_predictions=log_predictions)
        
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
        
        if log_predictions:
            statistics["train"].append(train_stats)
            
            # Add predictions from all the different sets / probes
            predictions[epoch] = {}  # Dict of dict
            predictions[epoch]["train"] = train_preds
            predictions[epoch]["test"] = test_preds
            predictions[epoch]["backdoor"] = backdoor_preds
            predictions[epoch]["backdoor_val"] = val_backdoor_preds
            predictions[epoch]["backdoor_reversed_val"] = val_backdoor_reversed_preds
            predictions[epoch]["backdoor_single_pix_val"] = val_backdoor_single_pix_preds
            predictions[epoch]["backdoor_reversed_single_pix_val"] = val_backdoor_reversed_single_pix_preds
            predictions[epoch]["backdoor_random_val"] = val_backdoor_random_preds
            predictions[epoch]["clean"] = backdoor_preds
            predictions[epoch]["clean_val"] = val_clean_preds
        
        statistics["test"].append(test_stats)
        statistics["clean"].append(clean_stats)
        statistics["clean_val"].append(val_clean_stats)
        statistics["backdoor"].append(backdoor_stats)
        statistics["backdoor_val"].append(val_backdoor_stats)
        statistics["backdoor_reversed_val"].append(val_backdoor_reversed_stats)
        statistics["backdoor_single_pix_val"].append(val_backdoor_single_pix_stats)
        statistics["backdoor_reversed_single_pix_val"].append(val_backdoor_reversed_single_pix_stats)
        statistics["backdoor_random_val"].append(val_backdoor_random_stats)
        
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


# In[ ]:


print("Final test accuracy:", statistics["test"][-1])


# In[ ]:


line_styles = ['solid', 'dashed', 'dashdot', 'dotted']
marker_list = ['o', '*', 'X', 'P', 'p', 'D', 'v', '^', 'h', '1', '2', '3', '4']
marker_colors = ["tab:gray", "tab:green", "tab:blue", "tab:purple", "tab:orange", "tab:red", "tab:pink", "tab:olive", "tab:brown", "tab:cyan"]

plot_train_test_sets = False
linewidth = 5.0
alpha = 0.7

for val_included in [True, False]:
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 6)
    
    x_vals = list(range(1, len(statistics["test"])+1))
    for idx, k in enumerate(natsort.natsorted(list(statistics.keys()))):
        if k == "predictions":
            continue
        if not log_predictions and k == "train":
            continue
        if not val_included and "_val" in k:
            continue
        if not plot_train_test_sets and ("train" in k or "test" in k):
            continue
        # line = plt.plot(x_vals, [x["acc"] for x in statistics[k]], linewidth=2., marker=marker_list[idx % len(marker_list)],
        #                 color=marker_colors[idx % len(marker_colors)], alpha=0.75, markeredgecolor='k', label=label_map_dict[k])
        line = plt.plot(x_vals, [x["acc"] for x in statistics[k]], linewidth=linewidth, color=marker_colors[idx % len(marker_colors)], 
                        alpha=alpha, label=label_map_dict[k])
        line[0].set_color(marker_colors[idx % len(marker_colors)])
        line[0].set_linestyle(line_styles[idx % len(line_styles)])

    plt.legend(prop={'size': font_size})
    plt.xlabel("Epochs", fontsize=font_size)
    plt.ylabel("Accuracy (%)", fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)

    if include_plot_title:
        plt.title(f"Training accuracy dynamics computed for ResNet-50 (CIFAR-100)", fontsize=font_size)
    plt.tight_layout()
    output_file = os.path.join(experiment_output_dir, f"probe_acc_{dataset}{'_val' if val_included else ''}.png")
    if main_proc and output_file is not None:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.show()


# In[ ]:


line_styles = ['solid', 'dashed', 'dashdot', 'dotted']
marker_list = ['o', '*', 'X', 'P', 'p', 'D', 'v', '^', 'h', '1', '2', '3', '4']
marker_colors = ["tab:gray", "tab:green", "tab:blue", "tab:purple", "tab:orange", "tab:red", "tab:pink", "tab:olive", "tab:brown", "tab:cyan"]

for val_included in [True, False]:
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 6)
    
    x_vals = list(range(1, len(statistics["test"])+1))
    for idx, k in enumerate(natsort.natsorted(list(statistics.keys()))):
        if k == "predictions":
            continue
        if not log_predictions and k == "train":
            continue
        if not val_included and "_val" in k:
            continue
        line = plt.plot(x_vals, [x["loss"] for x in statistics[k]], linewidth=linewidth, color=marker_colors[idx % len(marker_colors)], 
                        alpha=alpha, label=label_map_dict[k])
        line[0].set_color(marker_colors[idx % len(marker_colors)])
        line[0].set_linestyle(line_styles[idx % len(line_styles)])

    plt.legend(prop={'size': font_size})
    plt.xlabel("Epochs", fontsize=font_size)
    plt.ylabel("Loss", fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)

    if include_plot_title:
        plt.title(f"Training loss dynamics computed for ResNet-50 (CIFAR-100)", fontsize=font_size)
    plt.tight_layout()
    output_file = os.path.join(experiment_output_dir, f"probe_loss_{dataset}{'_val' if val_included else ''}.png")
    if main_proc and output_file is not None:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.show()


# In[ ]:


if not log_predictions:
    print("Can't compute other statistics without the model predictions...")
    exit()


# ### Learning dynamics per example


# In[ ]:


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
    epoch_cummulative_scores = {k: [] for k in unique_probe_identity}

    print("Computing cummulative statistics...")
    for epoch in tqdm(range(num_epochs)):
        examples_learned_at_epoch = epoch_learned == epoch
        learned_ex_idx = np.nonzero(examples_learned_at_epoch)[0]
        for i in learned_ex_idx:
            k = dataset_probe_identity[i]
            stats[k] += 1
        for k in unique_probe_identity:
            epoch_cummulative_scores[k].append(stats[k])

    print("Statistics:", stats)
    print("Cummulative stats:", epoch_cummulative_scores)
    total_examples_learned = 0
    for k in stats:
        total_examples_learned += stats[k]
    print("Total examples learned in the end:", total_examples_learned)
    assert total_examples_learned == (len(epoch_learned) - int(np.sum(epoch_learned == -1)))


    # In[ ]:


    # First learned stats
    stats_first_learned = {k: 0 for k in unique_probe_identity}
    epoch_cummulative_scores_first_learned = {k: [] for k in unique_probe_identity}

    print("Computing first-learned statistics...")
    for epoch in range(num_epochs):
        examples_learned_at_epoch = epoch_first_learned == epoch
        learned_ex_idx = np.nonzero(examples_learned_at_epoch)[0]
        for i in learned_ex_idx:
            k = dataset_probe_identity[i]
            stats_first_learned[k] += 1
        for k in unique_probe_identity:
            epoch_cummulative_scores_first_learned[k].append(stats_first_learned[k])

    print("Statistics:", stats_first_learned)
    print("Cummulative stats:", epoch_cummulative_scores_first_learned)
    total_examples_learned = 0
    for k in stats:
        total_examples_learned += stats_first_learned[k]
    print("Total examples learned at any point during training:", total_examples_learned)
    assert total_examples_learned == (len(epoch_first_learned) - int(np.sum(epoch_first_learned == -1)))

    if main_proc:
        # Save the final statistics
        with open(data_statistics_file, "wb") as f:
            final_statistics = [sorted_ex_list, epoch_learned, epoch_first_learned, stats, epoch_cummulative_scores, stats_first_learned, epoch_cummulative_scores_first_learned]
            pickle.dump(final_statistics, f, protocol=pickle.HIGHEST_PROTOCOL)
else:
    assert os.path.exists(data_statistics_file)
    print("Data files already found. Loading data statistics from file:", data_statistics_file)
    
    with open(data_statistics_file, "rb") as f:
        final_statistics = pickle.load(f)
        sorted_ex_list, epoch_learned, epoch_first_learned, stats, epoch_cummulative_scores, stats_first_learned, epoch_cummulative_scores_first_learned = final_statistics


# In[ ]:


# Normalization should only happen for num_train_probes (val probes are separate)
normalizers = {k: (num_train_probes if k != "train" else (len(train_set) - len(discarded_idx))) for k in unique_probe_identity}
print("Normalizers:", normalizers)


# In[ ]:


for val_included in [True, False]:
    for iden, epoch_scores in enumerate([epoch_cummulative_scores, epoch_cummulative_scores_first_learned]):
        fig, ax = plt.subplots()
        fig.set_size_inches(8, 6)

        for idx, k in enumerate(natsort.natsorted(list(epoch_scores.keys()))):
            if not val_included and "_val" in k:
                continue
            if not plot_train_test_sets and ("train" in k or "test" in k):
                continue
            
            y = epoch_scores[k]
            x = np.arange(len(y))
            y_norm = [(float(i) / normalizers[k]) * 100. for i in y]
            # line = plt.plot(x, y_norm, linewidth=2., marker=marker_list[idx % len(marker_list)],
            #                 color=marker_colors[idx % len(marker_colors)], alpha=0.75, markeredgecolor='k', label=label_map_dict[k])
            line = plt.plot(x_vals, y_norm, linewidth=linewidth, color=marker_colors[idx % len(marker_colors)], 
                            alpha=alpha, label=label_map_dict[k])
            line[0].set_color(marker_colors[idx % len(marker_colors)])
            line[0].set_linestyle(line_styles[idx % len(line_styles)])

        plt.xlabel("Number of epochs", fontsize=font_size)
        # plt.ylabel(f"Fraction of examples learned{'at any point during training' if iden == 1 else ''} (%)", fontsize=font_size)
        plt.ylabel(f"Fraction of examples learned (%)", fontsize=font_size)
        if include_plot_title:
            plt.title(f"Learning dynamics computed for ResNet-50 (CIFAR-100)", fontsize=font_size)
        plt.legend(prop={'size': font_size})
        plt.ylim(0., 100.)
        
        plt.xticks(fontsize=font_size)
        plt.yticks(fontsize=font_size)

        plt.tight_layout()
        output_file = os.path.join(experiment_output_dir, f"{'first_learned' if iden == 1 else 'learning'}_dynamics_{dataset}{'_val' if val_included else ''}.png")
        if main_proc and output_file is not None:
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close('all')


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


# In[ ]:


class_names = list(np.unique(dataset_probe_identity))
print(class_names)


# In[ ]:


fig, ax = plt.subplots(1, 1, figsize=(50, 10))
labels = list(range(1, len(sorted_losses_all)+1))
color_list = ['tab:red', 'tab:blue', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:cyan', 'tab:olive', 'tab:gray']
plot_points = False

handles = []
legend_label = []
for i, cls in enumerate(class_names):
    if cls in ["train", "train_noisy"]:
        continue
    print("Class:", cls)
    color = color_list[i % len(color_list)]
    patch = mpatches.Patch(color=color)
    handles.append(patch)
    # legend_label.append(cls.replace("_", " ").title())
    legend_label.append(label_map_dict[cls])
    
    data = []
    for epoch in range(len(sorted_losses_all)):
        current_losses = [float(sorted_losses_all[epoch][i]) for i in range(len(sorted_losses_all[epoch])) if str(dataset_probe_identity[i]) == cls and sorted_losses_all[epoch][i] is not None]
        data.append(current_losses)
    
    # parts = ax.boxplot(data, notch=True, patch_artist=True, showfliers=False)
    parts = ax.boxplot(data, notch=True, patch_artist=True, showfliers=False,
                       boxprops=dict(facecolor=color, color=color, alpha=0.3),
                       capprops=dict(color=color),
                       whiskerprops=dict(color=color),
                       flierprops=dict(color=color, markeredgecolor=color),
                       medianprops=dict(color=color))
    
    if plot_points:
        raise NotImplementedError
        # Plot the points
        num_points = 50
        for i in range(len(related_items)):
            ax.scatter([i+1 for _ in range(num_points)], np.random.choice(data[(related_items[i]*2)+(0 if diagonal else 1)], num_points), alpha=0.1, color=color)

ax.legend(handles, legend_label, prop={'size': font_size})
plt.ylabel("Loss values", fontsize=font_size)
plt.xlabel("Epochs", fontsize=font_size)
# plt.ylim(0, 6)

plt.tight_layout()
output_file = os.path.join(experiment_output_dir, f"loss_dist_{dataset}.png")
if main_proc and output_file is not None:
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
plt.show()


# In[ ]:


loss_dynamics_output_dir = os.path.join(experiment_output_dir, "loss_distribution")
violin_loss_dynamics_output_dir = os.path.join(experiment_output_dir, "loss_distribution_violin")
if main_proc and not os.path.exists(loss_dynamics_output_dir):
    os.mkdir(loss_dynamics_output_dir)
if main_proc and not os.path.exists(violin_loss_dynamics_output_dir):
    os.mkdir(violin_loss_dynamics_output_dir)


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
    plt.show()
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
    plt.show()
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
    train_set.transform = transforms.Compose(no_transform)
    if "cifar" in dataset_name or dataset_name == "mnist":
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

