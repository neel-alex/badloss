import argparse


def config():
    parser = argparse.ArgumentParser(description="Run a backdoor experiment")

    # Main arguments
    parser.add_argument('--dataset', default='cifar10', type=str,
                        choices=['cifar10', 'gtsrb', 'imagenette'])
    parser.add_argument('--attack', default='all', type=str,
                        choices=['all', 'patch', 'single_pix', 'random',
                                 'fixed', 'sinusoid', 'narcissus', 'frequency',
                                 'warped'])
    parser.add_argument('--defense', default='badloss', type=str,
                        choices=['badloss', 'nc', 'ac', 'ss', 'freq', 'abl',
                                 'cd', 'cbd', 'pss'])
    parser.add_argument('--poisoning_ratio', default=None, type=float)

    # Run arguments
    parser.add_argument('--retrain', default=False, type=bool)
    parser.add_argument('--num_epochs', default=None, type=int)
    parser.add_argument('--arch', default='resnet50', type=str,
                        choices=['resnet50', 'resnet18', 'vgg16',
                                 'densenet', 'squeezenet', 'efficientnet',
                                 'resnet34'])

    # Defense arguments
    parser.add_argument('--num_train_probes', default=250, type=int)
    parser.add_argument('--badloss_pretrain_epochs', default=30, type=int)

    # Defense-specific arguments
    parser.add_argument('--cbd_ce_gamma', default=1.0, type=float)

    return parser
