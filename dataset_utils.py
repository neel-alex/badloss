import os
import urllib

from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10, CIFAR100, GTSRB, ImageFolder


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
