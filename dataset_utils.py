import os
import random
import urllib
import copy
import itertools

import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10, GTSRB, ImageFolder, Imagenette
try:
    from catalyst.data import DistributedSamplerWrapper
except ImportError:
    print("catalyst not found for DistributedSamplerWrapper!")

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
    else:
        raise RuntimeError(f"Unknown dataset: {dataset}")
    name2label = {label2name[key]: key for key in label2name.keys()}
    assert name2label[label2name[0]] == 0
    return label2name, name2label


def get_settings_for_dataset(dataset, use_augmentations=True):
    data_dir = f"./data/{dataset}/"
    if dataset == "gtsrb":
        use_augmentations = False
    if dataset == "cifar10":
        img_size = (32, 32, 3)
        train_transform = [transforms.RandomHorizontalFlip(),
                           transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
                           transforms.ToTensor()]
        test_transform = [transforms.ToTensor()]
        if not use_augmentations:
            train_transform = test_transform
        no_transform = test_transform

        train_set = CIFAR10(data_dir, download=True, train=True, transform=transforms.Compose(train_transform))
        train_set_wo_aug = CIFAR10(data_dir, download=True, train=True, transform=transforms.Compose(no_transform))
        test_set = CIFAR10(data_dir, download=True, train=False, transform=transforms.Compose(test_transform))
    else:
        assert dataset == "imagenet" or dataset == "gtsrb" or dataset == "imagenette"
        img_size = (224, 224, 3)
        if dataset == "gtsrb":  # specifically for GTSRB
            rand_crop_scale = (0.8, 1.0)
        else:  # imagenet default
            rand_crop_scale = (0.08, 1.0)

        if use_augmentations:
            print("Training w/ augmentations...")
            train_transform = [transforms.RandomResizedCrop(224, scale=rand_crop_scale),
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
            train_set = GTSRB(data_dir, download=True, split="train", transform=transforms.Compose(train_transform))
            train_set_wo_aug = GTSRB(data_dir, download=True, split="train", transform=transforms.Compose(no_transform))
            test_set = GTSRB(data_dir, download=True, split="test", transform=transforms.Compose(test_transform))
            train_set.targets = [label for (img, label) in train_set]
            train_set_wo_aug.targets = [label for (img, label) in train_set_wo_aug]
            test_set.targets = [label for (img, label) in test_set]
        elif dataset == "imagenet":
            data_dir = "/ds/images/imagenet/"  # TODO: Configure dataset path
            train_set = ImageFolder(os.path.join(data_dir, "train"), transform=transforms.Compose(train_transform))
            train_set_wo_aug = ImageFolder(os.path.join(data_dir, "train"), transform=transforms.Compose(no_transform))
            test_set = ImageFolder(os.path.join(data_dir, "val_folders"), transform=transforms.Compose(test_transform))

            # Replace train_set.classes with real names
            train_set.original_classes = train_set.classes
        else:
            assert dataset == "imagenette"
            train_set = Imagenette(data_dir, download=False, split="train", transform=transforms.Compose(train_transform))
            train_set_wo_aug = Imagenette(data_dir, download=False, split="train", transform=transforms.Compose(no_transform))
            test_set = Imagenette(data_dir, download=False, split="val", transform=transforms.Compose(test_transform))
            
            train_set.targets = [label for (img, label) in train_set]
            train_set_wo_aug.targets = [label for (img, label) in train_set_wo_aug]
            test_set.targets = [label for (img, label) in test_set]


        label2name, _ = load_class_mapping(dataset)
        label2name = {k: v.split(',')[0][:20] for k, v in label2name.items()}
        train_set.classes = label2name  # Dict mapping from label to class name
    return img_size, train_transform, test_transform, no_transform, data_dir, train_set, train_set_wo_aug, test_set


class CustomTensorDataset(torch.utils.data.Dataset):
    def __init__(self, x: torch.Tensor, y: list, transform=None) -> None:
        self.x = x
        self.y = y
        self.transform = transform

    def __getitem__(self, index):
        if self.transform:
            return self.transform(self.x[index]), self.y[index]
        return self.x[index], self.y[index]

    def __len__(self):
        return self.x.size(0)


def seed_worker(worker_id: int) -> None:
    # Torch seeds each worker from the loader's generator; propagate that to numpy/random.
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_loader(dataset, distributed, num_workers, seed: int, indices=None, batch_size=16, shuffle=False):
    # Each loader owns its RNG, so data order doesn't depend on unrelated global RNG draws.
    generator = torch.Generator().manual_seed(seed)
    sampler = None
    if indices is not None:
        sampler = torch.utils.data.SubsetRandomSampler(indices, generator=generator)
    if distributed:
        if sampler is not None:
            print("Using distributed sampler on top of previous sampler...")
            sampler = DistributedSamplerWrapper(sampler)
        else:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset, seed=seed)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                         sampler=sampler, num_workers=num_workers, prefetch_factor=4,
                                         pin_memory=True, worker_init_fn=seed_worker, generator=generator)
    return loader


def make_probe_dataset(probes, train_set, dataset, defense, train_transform, val_probe_attacks, output_dir):
    discarded_idx = set(probes['all_backdoor_idx'])
    train_indices = [i for i in range(len(train_set)) if i not in discarded_idx]
    print("Discarded examples:", len(train_set) - len(train_indices))
    assert len(train_set) - len(train_indices) == len(discarded_idx)

    if defense == "badloss":
        probes_to_be_used = ["backdoor", "clean"]
        print("Selected probes to be used:", probes_to_be_used)

        probe_images = torch.cat([probes[k] for k in probes_to_be_used], dim=0)
        probe_labels = torch.cat([probes[f"{k}_labels"] for k in probes_to_be_used], dim=0)

        # Filter the train indexes
        probe_identity = list(itertools.chain(*([identity] * len(probes[identity])
                                                for identity in probes_to_be_used)))

        assert len(probe_identity) == len(probe_images), f"{len(probe_identity)} != {len(probe_images)}"

        print(f"Probe | Images: {probe_images.shape} | Labels: {probe_labels.shape}")

        probe_dataset = torch.utils.data.TensorDataset(probe_images, probe_labels)
        probe_dataset_standard = CustomTensorDataset(probe_images.to("cpu"),
                                                     [int(x) for x in probe_labels.to("cpu").numpy().tolist()],
                                                     transform=transforms.Compose(train_transform[:-1]))  # Cut off ToTensor transform
        # NB: indexing [0] applies the random train transform, which consumes global torch RNG
        print("Probe dataset:", len(probe_dataset_standard), probe_dataset_standard[0][0].shape,
              probe_dataset_standard[0][1])
        print("Curated probe dataset")
        plot(torch.stack([x[0] for x in probe_dataset], dim=0), torch.stack([x[1] for x in probe_dataset], dim=0),
             class_names=train_set.classes, output_file=f"probes_dataset_{dataset}.png", output_dir=output_dir)

    # TODO: Note how this adds "val".... hopefully this just solves problems and doesn't cause any lol
    val_probe_identity = list(itertools.chain(*([f"backdoor_{identity}_val"] * len(probes[f"backdoor_{identity}"])
                                                for identity in val_probe_attacks)))

    # Create the validation set for probes
    val_probe_images = torch.cat([probes[f"backdoor_{attack}"] for attack in val_probe_attacks], dim=0)
    val_probe_labels = torch.cat([probes[f"backdoor_{attack}_labels"] for attack in val_probe_attacks], dim=0)
    val_probe_dataset_standard = CustomTensorDataset(val_probe_images.to("cpu"),
                                                     [int(x) for x in val_probe_labels.to("cpu").numpy().tolist()],
                                                     transform=transforms.Compose(train_transform[:-1]))
    # NB: as above, this consumes global torch RNG
    print("Validation probe dataset:", len(val_probe_dataset_standard), val_probe_dataset_standard[0][0].shape,
          val_probe_dataset_standard[0][1])

    if defense == "badloss":
        comb_train_set = torch.utils.data.ConcatDataset([train_set, probe_dataset_standard, val_probe_dataset_standard])
        comb_train_indices = train_indices + [(len(train_set) + x) for x in
                                              range(len(probe_dataset_standard) + len(val_probe_dataset_standard))]
        dataset_probe_identity = ["train" for i in range(len(train_set))] + probe_identity + val_probe_identity
    else:
        comb_train_set = torch.utils.data.ConcatDataset([train_set, val_probe_dataset_standard])
        comb_train_indices = train_indices + [(len(train_set) + x) for x in
                                              range(len(val_probe_dataset_standard))]
        dataset_probe_identity = ["train" for i in range(len(train_set))] + val_probe_identity

    print("Indices in combined dataset:", len(comb_train_indices))
    assert len(np.unique(comb_train_indices)) == len(comb_train_indices)
    print("Size of combined dataset:", len(comb_train_set))

    assert len(dataset_probe_identity) == len(comb_train_set), f"{len(dataset_probe_identity)} != {len(comb_train_set)}"

    return comb_train_set, comb_train_indices, dataset_probe_identity, discarded_idx


class IdxDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx], idx


def make_index_dataset(comb_train_set, comb_train_indices, test_set,
                       no_transform, batch_size, distributed, num_workers, seed: int):
    # Convert into a dataset which returns indices
    idx_dataset = IdxDataset(comb_train_set)

    # Update dataset transform for evaluation | idx dataset -> concate dataset -> actual training dataset
    idx_dataset_wo_aug = copy.deepcopy(idx_dataset)
    idx_dataset_wo_aug.dataset.datasets[0].transform = transforms.Compose(no_transform)

    new_idx_loader = get_loader(idx_dataset, distributed, num_workers, seed,
                                indices=comb_train_indices, batch_size=batch_size)
    new_idx_loader_wo_aug = get_loader(idx_dataset_wo_aug, distributed, num_workers, seed,
                                       indices=comb_train_indices, batch_size=batch_size)
    test_idx_loader = get_loader(IdxDataset(test_set), distributed, num_workers, seed, batch_size=batch_size)

    return new_idx_loader, new_idx_loader_wo_aug, test_idx_loader, idx_dataset

