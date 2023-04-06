import os
import urllib
import natsort
import copy

import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10, CIFAR100, GTSRB, ImageFolder
from catalyst.data import DistributedSamplerWrapper

from plot_utils import plot


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
    elif dataset == "gtsrb":
        classes = ["Speed limit (20km/h)", "Speed limit (30km/h)", "Speed limit (50km/h)", "Speed limit (60km/h)", "Speed limit (70km/h)", "Speed limit (80km/h)",
                   "End of speed limit (80km/h)", "Speed limit (100km/h)", "Speed limit (120km/h)", "No passing", "No passing veh over 3.5 tons",
                   "Right-of-way at intersection", "Priority road", "Yield", "Stop", "No vehicles", "Veh > 3.5 tons prohibited", "No entry", "General caution",
                   "Dangerous curve left", "Dangerous curve right", "Double curve", "Bumpy road", "Slippery road", "Road narrows on the right", "Road work",
                   "Traffic signals", "Pedestrians", "Children crossing", "Bicycles crossing", "Beware of ice/snow", "Wild animals crossing",
                   "End speed + passing limits", "Turn right ahead", "Turn left ahead", "Ahead only", "Go straight or right", "Go straight or left",
                   "Keep right", "Keep left", "Roundabout mandatory", "End of no passing", "End no passing veh > 3.5 tons"]
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


def get_settings_for_dataset(dataset):
    if "mnist" in dataset:
        img_size = (28, 28, 1)
        train_transform = [transforms.ToTensor()]
        test_transform = [transforms.ToTensor()]
        no_transform = test_transform

        data_dir = f"./data/{dataset}/"  # TODO: Configure dataset path
        train_set = MNIST(data_dir, download=True, train=True, transform=transforms.Compose(train_transform))
        train_set_wo_aug = MNIST(data_dir, download=True, train=True, transform=transforms.Compose(no_transform))
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

        data_dir = f"/netscratch/siddiqui/Datasets/{dataset}/"  # TODO: Configure dataset path
        train_set = DatasetCls(data_dir, download=True, train=True, transform=transforms.Compose(train_transform))
        train_set_wo_aug = DatasetCls(data_dir, download=True, train=True, transform=transforms.Compose(no_transform))
        test_set = DatasetCls(data_dir, download=True, train=False, transform=transforms.Compose(test_transform))
    else:
        assert dataset == "imagenet" or dataset == "gtsrb"
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
        else:
            print("Training w/o augmentations...")
            train_transform = [transforms.Resize((224, 224)),
                               transforms.ToTensor()]
            test_transform = [transforms.Resize((224, 224)),
                              transforms.ToTensor()]
        no_transform = test_transform

        if dataset == "gtsrb":
            data_dir = f"/netscratch/siddiqui/Datasets/{dataset}/"  # TODO: Configure dataset path
            train_set = GTSRB(data_dir, download=True, split="train", transform=transforms.Compose(train_transform))
            train_set_wo_aug = GTSRB(data_dir, download=True, split="train", transform=transforms.Compose(no_transform))
            test_set = GTSRB(data_dir, download=True, split="test", transform=transforms.Compose(test_transform))
        else:
            assert dataset == "imagenet"

            data_dir = "/ds/images/imagenet/"  # TODO: Configure dataset path
            train_set = ImageFolder(os.path.join(data_dir, "train"), transform=transforms.Compose(train_transform))
            train_set_wo_aug = ImageFolder(os.path.join(data_dir, "train"), transform=transforms.Compose(no_transform))
            test_set = ImageFolder(os.path.join(data_dir, "val_folders"), transform=transforms.Compose(test_transform))

            # Replace train_set.classes with real names
            train_set.original_classes = train_set.classes

        label2name, _ = load_class_mapping(dataset)
        label2name = {k: v.split(',')[0][:20] for k, v in label2name.items()}
        train_set.classes = label2name  # Dict mapping from label to class name
    return img_size, train_transform, test_transform, no_transform, data_dir, train_set, train_set_wo_aug, test_set


class CustomTensorDataset(torch.utils.data.Dataset):
    def __init__(self, x: torch.Tensor, y: list) -> None:
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.x.size(0)


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


def get_loader(dataset, distributed, num_workers, indices=None, batch_size=16, shuffle=False):
    sampler = None
    if indices is not None:
        sampler = torch.utils.data.SubsetRandomSampler(indices)
    if distributed:
        if sampler is not None:
            print("Using distributed sampler on top of previous sampler...")
            sampler = DistributedSamplerWrapper(sampler)
        else:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                         sampler=sampler, num_workers=num_workers, pin_memory=True)
    return loader


def make_probe_dataset(probes, train_set, test_set, dataset, batch_size, num_example_probes, num_train_probes,
                       num_val_probes, attack_types, distributed, num_workers, output_dir, device):
    discarded_idx = list(probes["backdoor_idx"]) + list(probes["novel_backdoor_idx"]) + list(probes["clean_idx"])
    train_indices = [i for i in range(len(train_set)) if i not in discarded_idx]
    print("Discarded examples:", len(train_set) - len(train_indices))
    assert len(train_set) - len(train_indices) == len(discarded_idx)
    train_loader = get_loader(train_set, distributed, num_workers, train_indices, batch_size=batch_size)
    test_loader = get_loader(test_set, distributed, num_workers, batch_size=batch_size)

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

            val_probes[k] = torch.cat([probes[k][i:i + 1] for i in range(len(probes[k])) if i in val_idx],
                                      dim=0)  # Transfer val indices from train
            probes[k] = torch.cat([probes[k][i:i + 1] for i in range(len(probes[k])) if i not in val_idx],
                                  dim=0)  # Discard val index from train

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
    probe_dataset_standard = CustomTensorDataset(probe_images.to("cpu"),
                                                 [int(x) for x in probe_labels.to("cpu").numpy().tolist()])
    print("Probe dataset:", len(probe_dataset_standard), probe_dataset_standard[0][0].shape,
          probe_dataset_standard[0][1])

    # Create the validation set for probes
    val_probe_images = torch.cat([val_probes[k] for k in probes_to_be_used_val], dim=0)
    val_probe_labels = torch.cat([val_probes[f"{k}_labels"] for k in probes_to_be_used_val], dim=0)
    val_probe_dataset_standard = CustomTensorDataset(val_probe_images.to("cpu"),
                                                     [int(x) for x in val_probe_labels.to("cpu").numpy().tolist()])
    print("Validation probe dataset:", len(val_probe_dataset_standard), val_probe_dataset_standard[0][0].shape,
          val_probe_dataset_standard[0][1])

    # In[ ]:

    # Setup the probe validation set
    val_probe_dataset = ProbeDataset(val_probe_dataset_standard, val_probe_identity)
    val_probe_indices = [i for i in range(len(val_probe_dataset_standard))]
    val_probe_loader = get_loader(val_probe_dataset, distributed, num_workers, val_probe_indices, batch_size=batch_size)

    # In[ ]:

    print("Curated probe dataset")
    plot(torch.stack([x[0] for x in probe_dataset], dim=0), torch.stack([x[1] for x in probe_dataset], dim=0),
         class_names=train_set.classes, output_file=f"probes_dataset_{dataset}.png", output_dir=output_dir)

    return (probe_dataset_standard, val_probe_dataset_standard, val_probes,
            train_indices, probe_identity, val_probe_identity, discarded_idx)


class IdxDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx], idx


def make_index_dataset(comb_train_set, comb_train_indices, test_set,
                       no_transform, batch_size, distributed, num_workers):
    # Convert into a dataset which returns indices
    idx_dataset = IdxDataset(comb_train_set)

    # Update dataset transform for evaluation | idx dataset -> concate dataset -> actual training dataset
    idx_dataset_wo_aug = copy.deepcopy(idx_dataset)
    idx_dataset_wo_aug.dataset.datasets[0].transform = transforms.Compose(no_transform)

    new_idx_loader = get_loader(idx_dataset, distributed, num_workers, comb_train_indices, batch_size=batch_size)
    new_idx_loader_wo_aug = get_loader(idx_dataset_wo_aug, distributed, num_workers, comb_train_indices, batch_size=batch_size)
    test_idx_loader = get_loader(IdxDataset(test_set), distributed, num_workers, batch_size=batch_size)

    return new_idx_loader, new_idx_loader_wo_aug, test_idx_loader


def combine_dataset(train_set, train_indices, probe_dataset_standard, val_probe_dataset_standard,
                    probe_identity, val_probe_identity, use_val_probes_for_training):
    # Combine the two datasets (probe dataset and normal dataset)
    if use_val_probes_for_training:
        print("!! Including validation probes in the training process...")
        comb_train_set = torch.utils.data.ConcatDataset([train_set, probe_dataset_standard, val_probe_dataset_standard])
        comb_train_indices = train_indices + [(len(train_set) + x) for x in
                                              range(len(probe_dataset_standard) + len(val_probe_dataset_standard))]
    else:
        comb_train_set = torch.utils.data.ConcatDataset([train_set, probe_dataset_standard])
        comb_train_indices = train_indices + [(len(train_set) + x) for x in range(len(probe_dataset_standard))]
    print("Indices in combined dataset:", len(comb_train_indices))
    assert len(np.unique(comb_train_indices)) == len(comb_train_indices)
    print("Size of combined dataset:", len(comb_train_set))

    dataset_probe_identity = ["train" for i in range(len(train_set))] + probe_identity
    if use_val_probes_for_training:
        dataset_probe_identity += val_probe_identity
    assert len(dataset_probe_identity) == len(comb_train_set), f"{len(dataset_probe_identity)} != {len(comb_train_set)}"

    return comb_train_set, comb_train_indices, dataset_probe_identity
