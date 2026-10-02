import argparse
from typing import List, Dict
from collections import defaultdict


def config() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a backdoor experiment")

    # Main arguments
    parser.add_argument('--dataset', default='cifar10', type=str,
                        choices=['cifar10', 'gtsrb', 'imagenette', 'imagenet'])
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
    parser.add_argument('--wandb', action='store_true',
                        help="Log to Weights & Biases (off by default)")
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

    # Neural Cleanse
    parser.add_argument('--nc_cleanse_epochs', default=15, type=int)

    # Cognitive Distillation
    parser.add_argument('--cd_num_steps', default=100, type=int)

    # ABL
    parser.add_argument('--abl_pretrain_epochs', default=10, type=int)

    # CBD
    parser.add_argument('--cbd_ce_gamma', default=1.0, type=float)
    parser.add_argument('--cbd_pretrain_epochs', default=5, type=int)

    # PSS
    parser.add_argument('--pss_pretrain_epochs', default=2, type=int)
    parser.add_argument('--pss_intraclass_epochs', default=3, type=int)
    parser.add_argument('--pss_unlearn_epochs', default=20, type=int)

    return parser


def get_num_epochs(dataset):
    if dataset == 'cifar10':
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
        elif dataset == 'imagenette' or dataset == 'imagenet': # TODO
            return ['patch', 'random', 'fixed', 'sinusoid', 'frequency']
    else:
        return [attack]


def get_default_poisoning_ratio(dataset: str) -> Dict[str, float]:
    attack_ratios = {"patch": {'default': 0.01,
                               'gtsrb': 0.02,
                               'imagenette': 0.05,
                               'imagenet': 0.001},
                     "single_pix": {'default': 0.01,
                                    'gtsrb': 0.04},
                     "random": {'default': 0.01,
                                'imagenet': 0.001},
                     "fixed": {'default': 0.01,
                               'imagenet': 0.001},
                     "sinusoid": {'default': 0.1},  # frac. target class
                     "narcissus": {'default': 0.005},
                     "frequency": {'default': 0.01,
                                   'imagenet': 0.001}
                    }

    poisoning_ratios = {attack: ratios.get(dataset, ratios['default'])
                        for attack, ratios in attack_ratios.items()}

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
