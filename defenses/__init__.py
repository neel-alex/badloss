"""Each defense module exposes run(exp: experiment.Experiment)."""
import importlib

MODULES = {
    'badloss': 'badloss',
    'nc': 'neural_cleanse',
    'ac': 'activation_clustering',
    'ss': 'spectral_signatures',
    'freq': 'frequency',
    'abl': 'abl',
    'cd': 'cognitive_distillation',
    'cbd': 'cbd',
    'pss': 'pss',
}


def get_defense(name: str):
    return importlib.import_module(f"defenses.{MODULES[name]}")
