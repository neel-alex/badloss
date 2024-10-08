import argparse
from typing import List, Dict


def config() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a backdoor experiment")

    # Main arguments
    parser.add_argument('--dataset', default='cifar10', type=str,
                        choices=['cifar10', 'gtsrb', 'imagenette'])
    parser.add_argument('--attack', default='all', type=str,
                        choices=['all', 'patch', 'single_pix', 'random',
                                 'fixed', 'sinusoid', 'narcissus',
                                 'frequency'])
    parser.add_argument('--defense', default='badloss', type=str,
                        choices=['badloss', 'nc', 'ac', 'ss', 'freq', 'abl',
                                 'cd', 'cbd', 'pss'])
    parser.add_argument('--poisoning_ratio', nargs='*', default=[],
                        help="Poisoning ratios as key=value pairs, "
                             "e.g. patch=0.03")

    # Run arguments
    parser.add_argument('--seed', default=3, type=int)
    parser.add_argument('--retrain', default=False, type=bool)
    parser.add_argument('--num_epochs', default=None, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--arch', default='resnet50', type=str,
                        choices=['resnet50', 'resnet18', 'vgg16',
                                 'densenet', 'squeezenet', 'efficientnet',
                                 'resnet34'])

    # Defense arguments
    parser.add_argument('--num_train_probes', default=250, type=int)

    # Defense-specific arguments

    # BaDLoss
    parser.add_argument('--badloss_pretrain_epochs', default=30, type=int)
    parser.add_argument('--badloss_metric', default='loss', type=str,
                        choices=['loss', 'prob'])

    # CBD
    parser.add_argument('--cbd_ce_gamma', default=1.0, type=float)

    return parser


def get_num_epochs(dataset):
    if dataset == 'mnist':
        return 25
    elif dataset == 'cifar10':
        return 100
    elif dataset == 'gtsrb':
        return 100
    elif dataset == 'imagenette':
        return 250
    else:
        return 100


def get_attacks(attack: str, dataset: str) -> List[str]:
    if attack == 'all':
        if dataset == 'cifar10':
            return ['patch', 'single_pix', 'random', 'fixed',
                    'sinusoid', 'frequency', 'narcissus']
        elif dataset == 'gtsrb':
            return ['patch', 'single_pix', 'random', 'fixed',
                    'sinusoid', 'frequency']
        elif dataset == 'imagenette':
            return ['patch', 'random', 'fixed', 'sinusoid', 'frequency']
    else:
        return [attack]


def get_default_poisoning_ratio(dataset: str) -> Dict[str, float]:
    patch_ratios = {
        'cifar10': 0.01,
        'gtsrb': 0.02,
        'imagenette': 0.05,
    }
    single_pix_ratios = {
        'cifar10': 0.01,
        'gtsrb': 0.04,
    }

    poisoning_ratios = {
        "patch": patch_ratios[dataset],
        "single_pix": single_pix_ratios[dataset],
        "random": 0.01,
        "fixed": 0.01,
        "sinusoid": 0.1,  # Clean attacks are a fraction of the target class!
        "narcissus": 0.005,  # So they claim... 25 images!!
        "frequency": 0.01,  # They claim this is right, but it feels too high
    }

    return poisoning_ratios


def get_poisoning_ratio(dataset: str,
                        pois_ratios: List[str]) -> Dict[str, float]:
    default_ratios = get_default_poisoning_ratio(dataset)
    config_ratios = dict(pair.split('=') for pair in pois_ratios)
    # Turn into floats
    config_ratios = {k: float(v) for k, v in config_ratios.items()}
    # Combine, preferring values from the config
    combined_ratios = {**default_ratios, **config_ratios}
    return combined_ratios
