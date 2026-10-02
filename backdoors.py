import os
import collections

import numpy as np
import torch
from torchvision import transforms
import cv2

import dist_utils


RANDOM_BACKDOOR_ALPHA = 0.075
FIXED_BACKDOOR_ALPHA = 0.025
SINUSOID_BACKDOOR_ALPHA = 0.075
SINUSOID_BACKDOOR_FREQ = 6
IMAGENETTE_ALPHA = 0.15

BLENDING_ATTACKS = {'random', 'fixed', 'sinusoid', 'narcissus', 'frequency'}
BOOSTING_RATIO = 2
CLEAN_LABEL_ATTACKS = {'sinusoid', 'narcissus'}


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


def get_pattern(attack_name, img_size, dataset, output_dir, main_proc):
    pattern = None
    if attack_name == "random":
        pattern_file = os.path.join(output_dir, "random_pattern.png")
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
    elif attack_name == "fixed":
        pattern = np.zeros(img_size, dtype=np.float32)
        pattern[::2, ::2, :] = 1
        pattern = transforms.ToTensor()(pattern)
    elif attack_name == "sinusoid":
        pattern = np.zeros(img_size, dtype=np.float32)
        for col in range(pattern.shape[1]):
            pattern[:, col, :] = 1 - np.cos(2 * np.pi * col * SINUSOID_BACKDOOR_FREQ / pattern.shape[1])
        pattern = transforms.ToTensor()(pattern)
    elif attack_name == "narcissus":
        pattern = np.load(f'data/{attack_name}_noise_{dataset}.npy')
        pattern = torch.tensor(pattern[0])

    return pattern


class Identity(object):
    def __call__(self, x):
        return x


def make_probe_transform(attack_name, img_size, dataset, output_dir, main_proc, pattern=None, alpha_boost=1):
    """Returns (transform applying the attack's trigger, trigger pattern). `pattern` reuses an existing
    random pattern (so test-time triggers match the poisoned training images)."""
    if pattern is None:
        pattern = get_pattern(attack_name, img_size, dataset, output_dir, main_proc)

    if attack_name == "patch":
        backdoor = BackdoorPatch(imagenette=(dataset == 'imagenette'))
    elif attack_name == "single_pix":
        backdoor = BackdoorPatch(single_pixel_backdoor=True, imagenette=(dataset == 'imagenette'))
    elif attack_name in {"random", "fixed", "sinusoid"}:
        alpha = {"random": RANDOM_BACKDOOR_ALPHA, "fixed": FIXED_BACKDOOR_ALPHA,
                 "sinusoid": SINUSOID_BACKDOOR_ALPHA}[attack_name]
        if dataset == "imagenette":
            alpha = IMAGENETTE_ALPHA  # Stronger on imagenette
        backdoor = BackdoorPatch(pattern=pattern, alpha=alpha*alpha_boost)
    elif attack_name == "narcissus":
        backdoor = BackdoorPatch(pattern=pattern, alpha=1*alpha_boost, mode='add')
    elif attack_name == "frequency":
        magnitude = 90 if dataset == "imagenette" else 30  # Stronger on imagenette
        backdoor = FrequencyAttack(boost=alpha_boost, magnitude=magnitude)
    elif attack_name == "clean":
        backdoor = Identity()
    else:
        raise NameError(f"Attack type {attack_name} is not a valid attack type.")
    backdoor_transform = transforms.Compose([backdoor, ClampRangeTransform()])
    return backdoor_transform, pattern


def add_probe_data(probe_dict, key, indices, dataset, labels, transform=None, track_idx=True, compute_diffs=False):
    if track_idx:
        probe_dict[f"{key}_idx"] = indices
    if transform is not None:
        if track_idx:
            probe_dict[f"{key}_original"] = torch.stack([dataset[i][0] for i in indices], dim=0)
        probe_dict[f"{key}"] = torch.stack([transform(dataset[i][0]) for i in indices], dim=0)
    else:
        probe_dict[f"{key}"] = torch.stack([dataset[i][0] for i in indices], dim=0)
    probe_dict[f"{key}_labels"] = torch.from_numpy(labels)
    if compute_diffs and transform is not None and track_idx:
        probe_dict[f"{key}_diff"] = probe_dict[f"{key}_original"] - probe_dict[f"{key}"]


def make_train_probes(num_classes, dataset, train_set_wo_aug, num_train_probes, train_probe_attack, output_dir,
                      main_proc, img_size, val_probe_indices):
    probes = {}
    attack_target = np.random.choice(np.arange(num_classes))
    print("Chosen train probe target:", attack_target)
    train_indices = list(range(len(train_set_wo_aug)))
    valid_indices = [i for i in train_indices if i not in val_probe_indices]
    probe_indices = np.random.choice(valid_indices, size=2 * num_train_probes, replace=False)

    probe_transform, _ = make_probe_transform(train_probe_attack, img_size, dataset, output_dir, main_proc)
    backdoor_idx = probe_indices[:num_train_probes]
    if train_probe_attack == 'clean':
        attack_labels = np.array([train_set_wo_aug[i][1] for i in backdoor_idx])
    else:
        attack_labels = np.array([attack_target for i in backdoor_idx])
    add_probe_data(probes, "backdoor", backdoor_idx, train_set_wo_aug, attack_labels,
                   transform=probe_transform, compute_diffs=True)

    clean_idx = probe_indices[num_train_probes:]
    clean_labels = np.array([train_set_wo_aug[i][1] for i in clean_idx])
    add_probe_data(probes, "clean", clean_idx, train_set_wo_aug, clean_labels)

    probes['all_backdoor_idx'] = np.concatenate((probes['backdoor_idx'], probes['clean_idx']))
    return probes, attack_target


def make_val_probes(num_classes, dataset, train_set_wo_aug, num_val_probes, val_probe_attacks, output_dir,
                    main_proc, img_size):
    val_probes = {}
    attack_targets = {attack: np.random.choice(np.arange(num_classes)) for attack in val_probe_attacks}
    if 'sinusoid' in val_probe_attacks and dataset == "gtsrb":
        class_counts = collections.Counter(train_set_wo_aug.targets)
        while class_counts[attack_targets['sinusoid']] < 1000:
            attack_targets['sinusoid'] = np.random.choice(np.arange(num_classes))
    if 'narcissus' in val_probe_attacks:
        attack_targets['narcissus'] = narcissus_classes[dataset]
    print("Chosen val attack targets:", attack_targets)
    attack_numbers = {attack: int(len(train_set_wo_aug) * num_val_probes[attack]) for attack in val_probe_attacks}
    print("Making attack image quantities:", attack_numbers, "(clean label attacks may be incorrect)")
    train_indices = list(range(len(train_set_wo_aug)))
    random_pattern = None
    chosen_indices = np.array([], dtype=int)

    for attack in val_probe_attacks:
        target = attack_targets[attack]
        num = attack_numbers[attack]
        # For clean attacks, get clean indices to choose from.
        indices_to_choose_from = train_indices
        if attack in CLEAN_LABEL_ATTACKS:
            indices_to_choose_from = np.where(np.array(train_set_wo_aug.targets) == target)[0]
            # Clean label attacks are expressed as a fraction of the target class! Adjust attack number appropriately.
            num = int(num_val_probes[attack] * len(indices_to_choose_from))
            if dataset == "gtsrb":
                num = max(num, 300)

        # don't let multiple attacks hit the same image, including train probe images.
        indices_to_choose_from = [i for i in indices_to_choose_from if i not in chosen_indices]

        attack_idx = np.random.choice(indices_to_choose_from, size=min(num, len(indices_to_choose_from)), replace=False)
        attack_labels = np.array([target for _ in attack_idx])
        probe_transform, pattern = make_probe_transform(attack, img_size, dataset, output_dir, main_proc)
        add_probe_data(val_probes, f"backdoor_{attack}", attack_idx, train_set_wo_aug, attack_labels,
                       transform=probe_transform, compute_diffs=True)

        print(f"Backdoor ({attack}) probe shape:", val_probes[f"backdoor_{attack}"].shape)

        if attack == "random":
            random_pattern = pattern

        # update chosen_indices:
        chosen_indices = np.concatenate((chosen_indices, attack_idx))

    val_probes["all_backdoor_idx"] = chosen_indices

    return val_probes, attack_targets, random_pattern


def make_test_probes(test_set, dataset, num_test_probes, val_probe_attacks, attack_targets, random_pattern,
                     output_dir, main_proc, img_size):
    test_probes = {}

    for attack in val_probe_attacks:
        target = attack_targets[attack]

        pattern = random_pattern if attack == "random" else None
        probe_transform, _ = make_probe_transform(attack, img_size, dataset, output_dir, main_proc, pattern=pattern)

        # Don't use any clean indices -- this way, the attack success rate should start at 0.
        #   (though practically there will be some randomly classified training images with low test acc.)
        non_clean_test_indices = np.where(np.array(test_set.targets) != target)[0]
        test_indices = np.random.choice(non_clean_test_indices, size=min(len(non_clean_test_indices), num_test_probes),
                                        replace=False)
        test_labels = np.array([target for _ in test_indices])

        add_probe_data(test_probes, f"backdoor_{attack}", test_indices, test_set, test_labels,
                       transform=probe_transform, track_idx=False)
        if attack in BLENDING_ATTACKS:
            boosted_transform, _ = make_probe_transform(attack, img_size, dataset, output_dir, main_proc,
                                                        pattern=pattern, alpha_boost=BOOSTING_RATIO)
            add_probe_data(test_probes, f"backdoor_{attack}_boosted", test_indices, test_set, test_labels,
                           transform=boosted_transform, track_idx=False)

    return test_probes


narcissus_classes = {
    'cifar10': 2
}
