"""Frequency Analysis (Zeng et al., 2021), adapted to filter the training set and retrain.

Train a small CNN to separate the DCT spectra of clean images from those of synthetically corrupted copies.
For parity with BaDLoss, its training data comes only from the defender's bona fide clean probes. Examples it
classifies as corrupted are removed.
"""
import copy

import numpy as np
import torch
from scipy.fft import dct
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm


class FreqCNN(torch.nn.Module):
    def __init__(self, image_shape):
        """image_shape: [c, h, w]"""
        super().__init__()

        def block(c_in, c_out, dropout):
            return [torch.nn.Conv2d(c_in, c_out, kernel_size=3, padding=1), torch.nn.BatchNorm2d(c_out),
                    torch.nn.ELU(),
                    torch.nn.Conv2d(c_out, c_out, kernel_size=3, padding=1), torch.nn.BatchNorm2d(c_out),
                    torch.nn.ELU(),
                    torch.nn.MaxPool2d(kernel_size=2), torch.nn.Dropout2d(p=dropout)]

        self.layers = torch.nn.Sequential(
            *block(image_shape[0], 32, 0.2), *block(32, 64, 0.3), *block(64, 128, 0.4),
            torch.nn.Flatten(),
            torch.nn.Linear((image_shape[1] // 8) * (image_shape[2] // 8) * 128, 2))

    def forward(self, x):
        return self.layers(x)


def apply_random_transform(probe):
    """Synthetic backdoor-like corruption: a white or random-noise patch near a random corner.
    (The original detector also used noise, shadow and blending corruptions; restricted to patches here.)"""
    np.random.randint(0, 5)  # Vestigial draw, kept so the RNG stream is unchanged
    patch_x = np.random.randint(2, 8)
    patch_y = np.random.randint(2, 8)
    loc = np.random.randint(0, 6)
    corner = np.random.randint(0, 4)

    attack = np.random.randint(0, 2)
    if attack == 0:
        block = np.ones((3, patch_x, patch_y))
    else:
        block = np.random.rand(3, patch_x, patch_y)

    if corner == 0:
        probe[:, loc:loc+patch_x, loc:loc+patch_y] = block
    elif corner == 1:
        probe[:, loc:loc+patch_x, -(loc+patch_y):-loc or None] = block
    elif corner == 2:
        probe[:, -(loc+patch_x):-loc or None, loc:loc+patch_y] = block
    elif corner == 3:
        probe[:, -(loc+patch_x):-loc or None, -(loc+patch_y):-loc or None] = block

    return np.clip(probe, 0, 1)


def dct2(block):
    # From https://github.com/YiZeng623/frequency-backdoor/blob/main/Sec4_Frequency_Detection/Train_Detection.ipynb
    return dct(dct(block.T, norm='ortho').T, norm='ortho')


def to_dct(images):
    """In-place 2D DCT of each channel of an NCHW numpy array."""
    for n in range(images.shape[0]):
        for c in range(images.shape[1]):
            images[n, c, :, :] = dct2(images[n, c, :, :])
    return images


def get_indices_for_thresh_from_loader(exp, thresh, freq_model):
    freq_model.eval()
    identified_indices = []
    for (image, label), indices in tqdm(exp.new_idx_loader_wo_aug):
        image = torch.tensor(to_dct(image.cpu().numpy()), device=exp.device)
        probs = torch.nn.functional.softmax(freq_model(image), dim=1)[:, 1]
        identified_indices.append(indices[(probs.cpu() >= thresh).nonzero()[:, 0]])
    return torch.hstack(identified_indices)


def run(exp):
    args = exp.args
    exp.wandb_prefix = "retraining_"
    freq_probes = {
        'clean': exp.probe_imgs.images.cpu().numpy(),
        'backdoor': copy.deepcopy(exp.probe_imgs.images.cpu().numpy()),
    }
    for i in range(freq_probes['backdoor'].shape[0]):
        freq_probes['backdoor'][i] = apply_random_transform(freq_probes['backdoor'][i])
    for key in freq_probes:
        to_dct(freq_probes[key])

    freq_train_set = torch.vstack((torch.tensor(freq_probes['clean']), torch.tensor(freq_probes['backdoor'])))
    freq_labels = torch.hstack((torch.zeros(freq_probes['clean'].shape[0], dtype=torch.long),
                                torch.ones(freq_probes['backdoor'].shape[0], dtype=torch.long)))
    freq_dataset = TensorDataset(freq_train_set, freq_labels)
    freq_dataloader = DataLoader(freq_dataset, batch_size=32, shuffle=True,
                                 generator=torch.Generator().manual_seed(exp.seed))

    freq_model = FreqCNN(freq_train_set[0].shape).to(exp.device)
    freq_criterion = torch.nn.CrossEntropyLoss()
    freq_optimizer = torch.optim.Adadelta(freq_model.parameters(), lr=args.freq_lr, weight_decay=1e-4)

    for epoch in range(args.freq_epochs):
        epoch_loss = 0.
        epoch_correct = 0
        for batch, labels in freq_dataloader:
            batch, labels = batch.to(exp.device), labels.to(exp.device)
            freq_optimizer.zero_grad()
            outputs = freq_model(batch)
            loss = freq_criterion(outputs, labels)
            loss.backward()
            freq_optimizer.step()

            epoch_loss += loss.item()
            epoch_correct += (outputs.argmax(axis=1) == labels).sum().item()
        print(f"Epoch {epoch+1} loss: {epoch_loss/len(freq_dataset):.6f}, acc: {epoch_correct/len(freq_dataset):.6f}")

    auc_idx = [get_indices_for_thresh_from_loader(exp, t, freq_model) for t in np.linspace(0.1, 0.9, num=9)]
    exp.report_detection_auc("Freq", auc_idx)

    identified_indices = get_indices_for_thresh_from_loader(exp, args.freq_threshold, freq_model)
    exp.report_detection(identified_indices)
    exp.retrain(identified_indices, checkpoint_tag=args.freq_threshold)
