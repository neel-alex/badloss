import argparse
from typing import List, Dict, Optional


DATASETS = ['cifar10', 'gtsrb', 'imagenette', 'imagenet']
ATTACKS = ['patch', 'single_pix', 'blend_r', 'blend_p', 'sinusoid', 'narcissus', 'frequency']
DEFENSES = ['badloss', 'nc', 'ac', 'ss', 'freq', 'abl', 'cd', 'cbd', 'pss']


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a backdoor attack + defense experiment")

    # Main arguments
    parser.add_argument('--dataset', default='cifar10', type=str, choices=DATASETS)
    parser.add_argument('--attack', default='all', type=str, choices=['all'] + ATTACKS)
    parser.add_argument('--defense', default='badloss', type=str, choices=DEFENSES)
    parser.add_argument('--poisoning_ratio', nargs='*', default=[],
                        help="Poisoning ratios as key=value pairs, e.g. patch=0.03")

    # Run arguments
    parser.add_argument('--seed', default=3, type=int)
    parser.add_argument('--wandb', action='store_true', help="Log to Weights & Biases (off by default)")
    parser.add_argument('--data_dir', default='./data', type=str)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--num_epochs', default=None, type=int,
                        help="Epochs for (re)training models; default depends on the dataset")
    parser.add_argument('--batch_size', default=None, type=int,
                        help="Training batch size; default 128 for cifar10 (as in the paper), else 256")
    parser.add_argument('--eval_batch_size', default=None, type=int,
                        help="Batch size for evaluating in-memory probe sets; default depends on the dataset")
    parser.add_argument('--lr', default=1e-3, type=float, help="AdamW learning rate")
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--arch', default='resnet50', type=str,
                        choices=['resnet50', 'resnet18', 'resnet34', 'vgg16', 'densenet', 'squeezenet',
                                 'efficientnet'])
    parser.add_argument('--num_test_probes', default=None, type=int,
                        help="Triggered test images per attack for measuring ASR; default depends on the dataset")

    # Defense arguments
    parser.add_argument('--num_train_probes', default=250, type=int,
                        help="Number of bona fide clean examples (probes) available to the defender")

    # BaDLoss
    parser.add_argument('--badloss_pretrain_epochs', default=30, type=int)
    parser.add_argument('--badloss_retrain_epochs', default=None, type=int,
                        help="Epochs for retraining on the filtered set; default --num_epochs (50 for imagenet)")
    parser.add_argument('--badloss_metric', default='loss', type=str, choices=['loss', 'prob'],
                        help="Per-example quantity tracked over training: loss or correct-class probability")
    parser.add_argument('--badloss_k', default=50, type=int, help="Nearest clean trajectories used for scoring")
    parser.add_argument('--badloss_reject_frac', default=0.4, type=float,
                        help="Fraction of (non-probe) training examples removed as anomalous")
    parser.add_argument('--badloss_eps', default=0.01, type=float, help="Offset inside the log of the score")
    parser.add_argument('--badloss_no_epoch_filter', action='store_true',
                        help="Keep all epochs (default drops epochs with average-loss spikes)")
    parser.add_argument('--badloss_linear_scores', action='store_true',
                        help="Min-max normalize mean distances instead of the log score")
    parser.add_argument('--badloss_pretrain_augment', action='store_true',
                        help="Use data augmentation while collecting loss trajectories")

    # Neural Cleanse
    parser.add_argument('--nc_cleanse_epochs', default=15, type=int)
    parser.add_argument('--nc_anomaly_threshold', default=2.0, type=float, help="MAD anomaly index threshold")
    parser.add_argument('--nc_neuron_frac', default=0.01, type=float, help="Fraction of neurons treated as poisoned")
    parser.add_argument('--nc_fpr', default=0.05, type=float, help="Target false-positive rate on clean probes")

    # Activation Clustering
    parser.add_argument('--ac_mode', default='sil', type=str, choices=['sil', 'rsc'],
                        help="Silhouette score or relative cluster size")
    parser.add_argument('--ac_threshold', default=None, type=float,
                        help="Detection threshold; default 0.15 for sil, 0.3 for rsc")
    parser.add_argument('--ac_ica_components', default=10, type=int)

    # Spectral Signatures
    parser.add_argument('--ss_eps', default=0.1, type=float,
                        help="Assumed poisoning fraction; removes 1.5x this fraction of each class")

    # Frequency Analysis
    parser.add_argument('--freq_epochs', default=10, type=int)
    parser.add_argument('--freq_lr', default=0.05, type=float)
    parser.add_argument('--freq_threshold', default=0.5, type=float)

    # ABL
    parser.add_argument('--abl_pretrain_epochs', default=10, type=int)
    parser.add_argument('--abl_flooding', default=0.5, type=float)
    parser.add_argument('--abl_remove_frac', default=0.15, type=float)

    # Cognitive Distillation
    parser.add_argument('--cd_num_steps', default=100, type=int)
    parser.add_argument('--cd_remove_frac', default=0.15, type=float)

    # CBD
    parser.add_argument('--cbd_ce_gamma', default=1.0, type=float)
    parser.add_argument('--cbd_pretrain_epochs', default=5, type=int)
    parser.add_argument('--cbd_lr', default=0.1, type=float)
    parser.add_argument('--cbd_adv_lr', default=0.2, type=float)

    # PSS
    parser.add_argument('--pss_pretrain_epochs', default=2, type=int)
    parser.add_argument('--pss_intraclass_epochs', default=3, type=int)
    parser.add_argument('--pss_unlearn_epochs', default=20, type=int)
    parser.add_argument('--pss_lr', default=0.01, type=float)
    parser.add_argument('--pss_unlearn_lr', default=1e-4, type=float)
    parser.add_argument('--pss_clean_quantile', default=0.80, type=float,
                        help="Examples below this feature-consistency quantile are treated as clean")
    parser.add_argument('--pss_poison_quantile', default=0.95, type=float,
                        help="Examples above this feature-consistency quantile are treated as poisoned")

    return parser


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parses arguments and fills in dataset-dependent defaults, attack list and poisoning ratios."""
    args = build_parser().parse_args(argv)
    if args.batch_size is None:
        args.batch_size = 128 if args.dataset == 'cifar10' else 256
    defaults = dataset_defaults(args.dataset, args.batch_size)
    if args.num_epochs is not None:  # An explicit epoch count also applies to BaDLoss retraining
        defaults['badloss_retrain_epochs'] = args.num_epochs
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if args.ac_threshold is None:
        args.ac_threshold = 0.15 if args.ac_mode == 'sil' else 0.3
    args.attacks = get_attacks(args.attack, args.dataset)
    args.poison_ratios = get_poisoning_ratio(args.dataset, args.poisoning_ratio)
    return args


def dataset_defaults(dataset: str, batch_size: int) -> Dict:
    num_epochs = {'cifar10': 100, 'gtsrb': 100, 'imagenette': 250, 'imagenet': 100}[dataset]
    return {
        'num_epochs': num_epochs,
        'badloss_retrain_epochs': 50 if dataset == 'imagenet' else num_epochs,
        'eval_batch_size': 128 if dataset == 'cifar10' else batch_size,
        'num_test_probes': 2000 if dataset == 'imagenet' else 10000,
    }


def get_attacks(attack: str, dataset: str) -> List[str]:
    if attack == 'all':
        if dataset == 'cifar10':
            return ['patch', 'single_pix', 'blend_r', 'blend_p',
                    'sinusoid', 'frequency', 'narcissus']
        elif dataset == 'gtsrb':
            return ['patch', 'single_pix', 'blend_r', 'blend_p',
                    'sinusoid', 'frequency']
        elif dataset == 'imagenette' or dataset == 'imagenet':
            return ['patch', 'blend_r', 'blend_p', 'sinusoid', 'frequency']
    else:
        return [attack]


def get_default_poisoning_ratio(dataset: str) -> Dict[str, float]:
    attack_ratios = {"patch": {'default': 0.01,
                               'gtsrb': 0.02,
                               'imagenette': 0.05,
                               'imagenet': 0.001},
                     "single_pix": {'default': 0.01,
                                    'gtsrb': 0.04},
                     "blend_r": {'default': 0.01,
                                 'imagenet': 0.0005},
                     "blend_p": {'default': 0.01,
                                 'imagenet': 0.0002},
                     "sinusoid": {'default': 0.1},  # frac. target class
                     "narcissus": {'default': 0.005},  # frac. target class
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

