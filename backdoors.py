import os
import collections
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torchvision import transforms
import cv2

import dist_utils


BLEND_R_ALPHA = 0.075
BLEND_P_ALPHA = 0.025
SINUSOID_ALPHA = 0.075
SINUSOID_FREQ = 6
IMAGENETTE_ALPHA = 0.15

BLENDING_ATTACKS = {'blend_r', 'blend_p', 'sinusoid', 'narcissus', 'frequency'}
BOOSTING_RATIO = 2  # Trigger strength multiplier for the additional "_boosted" ASR evaluation
CLEAN_LABEL_ATTACKS = {'sinusoid', 'narcissus'}


@dataclass
class ImageSet:
    """A set of (possibly triggered) images with their training labels.

    idx: positions of the source images in their dataset; original/diff: the untriggered images and the
    trigger's effect (original - triggered), kept for plotting.
    """
    images: torch.Tensor
    labels: torch.Tensor
    idx: Optional[np.ndarray] = None
    original: Optional[torch.Tensor] = None
    diff: Optional[torch.Tensor] = None

    def __len__(self):
        return len(self.images)

    def to(self, device):
        self.images, self.labels = self.images.to(device), self.labels.to(device)
        return self


class BackdoorPatch(object):
    def __init__(self, single_pixel_backdoor=False, pattern=None, alpha=None, mode='average', imagenette=False):
        assert pattern is None or (not single_pixel_backdoor and alpha is not None)

        self.single_pixel_backdoor = single_pixel_backdoor

        self.pattern = pattern
        self.alpha = alpha
        self.mode = mode

        self.imagenette = imagenette

    def __call__(self, tensor):
        backdoor_pix_val = 1.0

        if self.pattern is not None and self.mode == 'average':
            tensor = (1 - self.alpha) * tensor + (self.alpha) * self.pattern
        elif self.pattern is not None and self.mode == 'add':
            tensor = tensor + self.pattern * self.alpha
        elif self.single_pixel_backdoor:
            if self.imagenette:
                tensor[:, 1, 1] = backdoor_pix_val
                tensor[:, 1, tensor.shape[2] - 2] = backdoor_pix_val
                tensor[:, tensor.shape[1] - 2, 1] = backdoor_pix_val
                tensor[:, tensor.shape[1] - 2, tensor.shape[2] - 2] = backdoor_pix_val
            else:
                tensor[:, tensor.shape[1] - 2, tensor.shape[2] - 2] = backdoor_pix_val
        else:
            if self.imagenette:
                tensor[:, 8:16, 8:16] = backdoor_pix_val
                tensor[:, 24:32, 8:16] = backdoor_pix_val
                tensor[:, 8:16, 24:32] = backdoor_pix_val
                tensor[:, 16:24, 16:24] = backdoor_pix_val
            else:
                tensor[:, tensor.shape[1] - 2, tensor.shape[2] - 2] = backdoor_pix_val
                tensor[:, tensor.shape[1] - 4, tensor.shape[2] - 2] = backdoor_pix_val
                tensor[:, tensor.shape[1] - 2, tensor.shape[2] - 4] = backdoor_pix_val
                tensor[:, tensor.shape[1] - 3, tensor.shape[2] - 3] = backdoor_pix_val
        return tensor


class ClampRangeTransform(object):
    def __init__(self):
        pass

    def __call__(self, x):
        return torch.clamp(x, 0., 1.)


class FrequencyAttack(object):
    def __init__(self, window_size=32, positions=((15, 15), (31, 31)), magnitude=30, boost=1):
        super().__init__()
        self.window_size = window_size
        self.positions = positions
        self.magnitude = magnitude * boost

    def __call__(self, x):
        x = x.numpy()
        x = np.moveaxis(x, 0, -1)
        x *= 255.
        x = x.astype(np.uint8)
        x = cv2.cvtColor(x, cv2.COLOR_RGB2YCrCb)
        x = np.moveaxis(x, -1, 0)

        x_dct = np.zeros(x.shape, dtype=float)
        for ch in range(x.shape[0]):
            for w in range(0, x.shape[1], self.window_size):
                for h in range(0, x.shape[2], self.window_size):
                    x_dct[ch, w:w+self.window_size, h:h+self.window_size] = cv2.dct(x[ch][w:w+self.window_size, h:h+self.window_size].astype(float))

        # Add trigger
        for ch in (1, 2):
            for w in range(0, x.shape[1], self.window_size):
                for h in range(0, x.shape[2], self.window_size):
                    for pos in self.positions:
                        x_dct[ch][w+pos[0], h+pos[1]] += self.magnitude

        for ch in range(x.shape[0]):
            for w in range(0, x.shape[1], self.window_size):
                for h in range(0, x.shape[2], self.window_size):
                    x[ch, w:w+self.window_size, h:h+self.window_size] = cv2.idct(x_dct[ch][w:w+self.window_size, h:h+self.window_size].astype(float))

        x = x.astype(np.uint8)
        x = np.moveaxis(x, 0, -1)
        x = cv2.cvtColor(x, cv2.COLOR_YCrCb2RGB)
        x = x / 255.
        x = np.clip(x, 0, 1)
        x = np.moveaxis(x, -1, 0)
        x = torch.tensor(x, dtype=torch.float)
        return x


def get_pattern(attack_name, img_size, dataset, output_dir, main_proc, data_root='./data'):
    pattern = None
    if attack_name == "blend_r":
        pattern_file = os.path.join(output_dir, "blend_r_pattern.png")
        if main_proc:
            pattern = np.clip(np.random.rand(*img_size) * 255, 0, 255)
            print("Random pattern shape:", pattern.shape)
            cv2.imwrite(pattern_file, pattern)
        dist_utils.wait_for_other_procs()  # Distributed barrier

        # Load the pattern to ensure the same pattern is loaded by all processes
        pattern_img = cv2.imread(pattern_file, cv2.IMREAD_UNCHANGED)
        pattern = transforms.ToTensor()(pattern_img)
        print(f"Random Pattern / Loaded shape: {pattern_img.shape} / Tensor shape: {pattern.shape} / "
              f"Min: {pattern.min()} / Max: {pattern.max()}")
    elif attack_name == "blend_p":
        pattern = np.zeros(img_size, dtype=np.float32)
        pattern[::2, ::2, :] = 1
        pattern = transforms.ToTensor()(pattern)
    elif attack_name == "sinusoid":
        pattern = np.zeros(img_size, dtype=np.float32)
        for col in range(pattern.shape[1]):
            pattern[:, col, :] = 1 - np.cos(2 * np.pi * col * SINUSOID_FREQ / pattern.shape[1])
        pattern = transforms.ToTensor()(pattern)
    elif attack_name == "narcissus":
        pattern = np.load(os.path.join(data_root, f'{attack_name}_noise_{dataset}.npy'))
        pattern = torch.tensor(pattern[0])

    return pattern


def make_probe_transform(attack_name, img_size, dataset, output_dir, main_proc, pattern=None, alpha_boost=1,
                         data_root='./data'):
    """Returns (transform applying the attack's trigger, trigger pattern). `pattern` reuses an existing
    blend_r pattern (so test-time triggers match the poisoned training images)."""
    if pattern is None:
        pattern = get_pattern(attack_name, img_size, dataset, output_dir, main_proc, data_root=data_root)

    if attack_name == "patch":
        backdoor = BackdoorPatch(imagenette=(dataset in {'imagenette', 'imagenet'}))
    elif attack_name == "single_pix":
        backdoor = BackdoorPatch(single_pixel_backdoor=True, imagenette=(dataset == 'imagenette'))
    elif attack_name in {"blend_r", "blend_p", "sinusoid"}:
        alpha = {"blend_r": BLEND_R_ALPHA, "blend_p": BLEND_P_ALPHA, "sinusoid": SINUSOID_ALPHA}[attack_name]
        if dataset in {"imagenette", "imagenet"}:
            alpha = IMAGENETTE_ALPHA  # Stronger on large images
        backdoor = BackdoorPatch(pattern=pattern, alpha=alpha*alpha_boost)
    elif attack_name == "narcissus":
        backdoor = BackdoorPatch(pattern=pattern, alpha=1*alpha_boost, mode='add')
    elif attack_name == "frequency":
        magnitude = 90 if dataset in {"imagenette", "imagenet"} else 30  # Stronger on large images
        backdoor = FrequencyAttack(boost=alpha_boost, magnitude=magnitude)
    else:
        raise NameError(f"Attack type {attack_name} is not a valid attack type.")
    backdoor_transform = transforms.Compose([backdoor, ClampRangeTransform()])
    return backdoor_transform, pattern


def make_image_set(indices, dataset, labels, transform=None, track_idx=True):
    """Stack dataset[i] for the given indices, optionally applying a trigger transform."""
    originals = torch.stack([dataset[i][0] for i in indices], dim=0) if transform is None or track_idx else None
    if transform is None:
        images = originals
    else:
        images = torch.stack([transform(dataset[i][0]) for i in indices], dim=0)
    image_set = ImageSet(images, torch.from_numpy(labels), idx=indices if track_idx else None)
    if transform is not None and track_idx:
        image_set.original = originals
        image_set.diff = originals - images
    return image_set


def make_probe_imgs(train_set_wo_aug, num_probes, poison_indices):
    """The defender's bona fide clean probes: num_probes clean training examples (not used by any poison)."""
    valid_indices = [i for i in range(len(train_set_wo_aug)) if i not in poison_indices]
    probe_idx = np.random.choice(valid_indices, size=num_probes, replace=False)
    probe_labels = np.array([train_set_wo_aug[i][1] for i in probe_idx])
    return make_image_set(probe_idx, train_set_wo_aug, probe_labels)


def make_poison_imgs(num_classes, dataset, train_set_wo_aug, poison_ratios, attacks, output_dir, main_proc,
                     img_size, data_root='./data'):
    """Poisoned training images for each attack (no image is attacked twice).

    Returns (poison_imgs: attack -> ImageSet, attack_targets: attack -> target class, blend_r pattern).
    """
    poison_imgs = {}
    attack_targets = {attack: np.random.choice(np.arange(num_classes)) for attack in attacks}
    if 'sinusoid' in attacks and dataset == "gtsrb":
        class_counts = collections.Counter(train_set_wo_aug.targets)
        while class_counts[attack_targets['sinusoid']] < 1000:
            attack_targets['sinusoid'] = np.random.choice(np.arange(num_classes))
    if 'narcissus' in attacks:
        attack_targets['narcissus'] = narcissus_classes[dataset]
    print("Chosen attack targets:", attack_targets)
    attack_numbers = {attack: int(len(train_set_wo_aug) * poison_ratios[attack]) for attack in attacks}
    print("Making attack image quantities:", attack_numbers, "(clean label attacks may be incorrect)")
    train_indices = list(range(len(train_set_wo_aug)))
    blend_r_pattern = None
    chosen_indices = np.array([], dtype=int)

    for attack in attacks:
        target = attack_targets[attack]
        num = attack_numbers[attack]
        indices_to_choose_from = train_indices
        if attack in CLEAN_LABEL_ATTACKS:
            # Clean-label attacks poison a fraction of the target class
            indices_to_choose_from = np.where(np.array(train_set_wo_aug.targets) == target)[0]
            num = int(poison_ratios[attack] * len(indices_to_choose_from))
            if dataset == "gtsrb":
                num = max(num, 300)

        indices_to_choose_from = [i for i in indices_to_choose_from if i not in chosen_indices]
        attack_idx = np.random.choice(indices_to_choose_from, size=min(num, len(indices_to_choose_from)),
                                      replace=False)
        attack_labels = np.array([target for _ in attack_idx])
        poison_transform, pattern = make_probe_transform(attack, img_size, dataset, output_dir, main_proc,
                                                         data_root=data_root)
        poison_imgs[attack] = make_image_set(attack_idx, train_set_wo_aug, attack_labels, transform=poison_transform)
        print(f"Poison ({attack}) shape:", poison_imgs[attack].images.shape)

        if attack == "blend_r":
            blend_r_pattern = pattern
        chosen_indices = np.concatenate((chosen_indices, attack_idx))

    return poison_imgs, attack_targets, blend_r_pattern


def all_poison_indices(poison_imgs):
    return np.concatenate([np.array([], dtype=int)] + [s.idx for s in poison_imgs.values()])


def make_poison_imgs_test(test_set, dataset, num_test_images, attacks, attack_targets, blend_r_pattern,
                          output_dir, main_proc, img_size, data_root='./data'):
    """Triggered test images (excluding each attack's target class), labeled with the attack's target, for
    measuring attack success rates. Blending attacks also get a "_boosted" set with a stronger trigger."""
    poison_imgs_test = {}
    for attack in attacks:
        target = attack_targets[attack]
        pattern = blend_r_pattern if attack == "blend_r" else None
        poison_transform, _ = make_probe_transform(attack, img_size, dataset, output_dir, main_proc, pattern=pattern,
                                                   data_root=data_root)

        non_target_indices = np.where(np.array(test_set.targets) != target)[0]
        test_indices = np.random.choice(non_target_indices, size=min(len(non_target_indices), num_test_images),
                                        replace=False)
        test_labels = np.array([target for _ in test_indices])

        poison_imgs_test[attack] = make_image_set(test_indices, test_set, test_labels, transform=poison_transform,
                                                  track_idx=False)
        if attack in BLENDING_ATTACKS:
            boosted_transform, _ = make_probe_transform(attack, img_size, dataset, output_dir, main_proc,
                                                        pattern=pattern, alpha_boost=BOOSTING_RATIO,
                                                        data_root=data_root)
            poison_imgs_test[f"{attack}_boosted"] = make_image_set(test_indices, test_set, test_labels,
                                                                   transform=boosted_transform, track_idx=False)
    return poison_imgs_test


narcissus_classes = {
    'cifar10': 2
}
