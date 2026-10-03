import torch


def classifier_layer(model):
    """The model's final linear layer (or, if it has none, e.g. SqueezeNet, its last top-level module); its input
    is used as the penultimate feature vector."""
    module = model.module if hasattr(model, 'module') else model  # DDP
    linears = [m for m in module.modules() if isinstance(m, torch.nn.Linear)]
    return linears[-1] if linears else list(module.children())[-1]


class FeatureExtractor:
    """Wraps a model so calls return (logits, penultimate features). Call remove() when done."""
    def __init__(self, model):
        self.model = model
        self.last_layer_activations = None
        self.handle = classifier_layer(model).register_forward_hook(self.hook)

    def hook(self, module, input, output):
        self.last_layer_activations = input[0]

    def __call__(self, x):
        logits = self.model(x)
        return logits, torch.flatten(self.last_layer_activations, 1)

    def remove(self):
        self.handle.remove()
