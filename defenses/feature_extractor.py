import torch


class FeatureExtractor:
    """Wraps a ResNet so calls return (logits, flattened avgpool features). Call remove() when done."""
    def __init__(self, model):
        self.model = model
        self.last_layer_activations = None
        self.handle = self.model.avgpool.register_forward_hook(self.hook)

    def hook(self, module, input, output):
        self.last_layer_activations = output

    def __call__(self, x):
        logits = self.model(x)
        return logits, torch.flatten(self.last_layer_activations, 1)

    def remove(self):
        self.handle.remove()
