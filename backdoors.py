import os

import numpy as np
import torch
from torchvision import transforms
import cv2
from PIL import Image

import dist_utils


RANDOM_BACKDOOR_ALPHA = 0.1
FIXED_BACKDOOR_ALPHA = 0.025
SINUSOID_BACKDOOR_ALPHA = 0.025
SINUSOID_BACKDOOR_FREQ = 6

CLEAN_LABEL_ATTACKS = {'sinusoid'}


class BackdoorPatch(object):
    def __init__(self, single_pixel_backdoor=False, reverse_backdoor=False,
                 pattern=None, alpha=None):
        assert pattern is None or (not single_pixel_backdoor and not reverse_backdoor and alpha is not None)

        self.single_pixel_backdoor = single_pixel_backdoor
        self.reverse_backdoor = reverse_backdoor

        self.pattern = pattern
        self.alpha = alpha

    def __call__(self, tensor):
        backdoor_pix_val = 1.0

        if self.pattern is not None:
            tensor = (1 - self.alpha) * tensor + (self.alpha) * self.pattern
        elif self.single_pixel_backdoor:
            if self.reverse_backdoor:
                tensor[:, 1, 1] = backdoor_pix_val
            else:
                tensor[:, tensor.shape[1] - 2, tensor.shape[2] - 2] = backdoor_pix_val
        else:
            if self.reverse_backdoor:
                tensor[:, 1, 1] = backdoor_pix_val
                tensor[:, 3, 1] = backdoor_pix_val
                tensor[:, 1, 3] = backdoor_pix_val
                tensor[:, 2, 2] = backdoor_pix_val
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


class WarpingAttack(object):
    def __init__(self, img_size, s=0.5, k=4, grid_rescale=1.0, identity_grid=None, noise_grid=None):
        self.img_size = img_size
        self.s = s
        self.k = k
        self.grid_rescale = grid_rescale

        if identity_grid is not None:
            self.identity_grid = identity_grid
        else:
            self.identity_grid = self.create_identity_grid(img_size)

        if noise_grid is not None:
            self.noise_grid = noise_grid
        else:
            self.noise_grid = self.generate_noise_grid(k, s, img_size, img_size)

    def create_identity_grid(self, img_size):
        x = np.linspace(-1, 1, img_size)
        y = np.linspace(-1, 1, img_size)
        x_t, y_t = np.meshgrid(x, y)
        identity_grid = np.stack((x_t, y_t), axis=2)
        return identity_grid.astype(np.float32)

    def generate_noise_grid(self, k, s, h, w):
        P = np.random.uniform(-1, 1, (k, k, 2))
        P /= np.mean(np.abs(P))
        P *= s
        M0 = cv2.resize(P, (h, w), interpolation=cv2.INTER_CUBIC)
        M = np.clip(M0, -1, 1)
        return M.astype(np.float32)

    def apply_warping(self, x, grid):
        x = torch.nn.functional.grid_sample(x.unsqueeze(0), grid.unsqueeze(0), align_corners=True)
        return x.squeeze(0)

    def __call__(self, x):
        grid_temps = (self.identity_grid + self.s * self.noise_grid / self.img_size) * self.grid_rescale
        grid_temps = torch.clamp(torch.tensor(grid_temps), -1, 1)
        warped_x = self.apply_warping(x, grid_temps)
        return warped_x


def get_pattern(attack_name, img_size, output_dir, main_proc):
    pattern = None
    if "random" in attack_name:
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
    if "fixed" in attack_name:
        pattern = np.zeros(img_size, dtype=np.float32)
        if "reversed" in attack_name:
            pattern[1::2, 1::2, :] = 1
        else:
            pattern[::2, ::2, :] = 1
        pattern = transforms.ToTensor()(pattern)
    if "sinusoid" in attack_name:
        pattern = np.zeros(img_size, dtype=np.float32)
        if "reversed" in attack_name:
            for row in range(pattern.shape[0]):
                pattern[row, :, :] = np.sin(2 * np.pi * row * SINUSOID_BACKDOOR_FREQ / pattern.shape[0])
        else:
            for col in range(pattern.shape[1]):
                pattern[:, col, :] = np.sin(2 * np.pi * col * SINUSOID_BACKDOOR_FREQ / pattern.shape[1])
        pattern = transforms.ToTensor()(pattern)

    return pattern


def make_probe_transform(attack_name, img_size, output_dir, main_proc, aux_data=None):
    if "random" in attack_name and aux_data is not None:
        pattern = aux_data
    else:
        pattern = get_pattern(attack_name, img_size, output_dir, main_proc)

    if attack_name == "patch":
        backdoor = BackdoorPatch()
    elif attack_name == "reversed_patch":
        backdoor = BackdoorPatch(reverse_backdoor=True)
    elif attack_name == "single_pix":
        backdoor = BackdoorPatch(single_pixel_backdoor=True)
    elif attack_name == "reversed_single_pix":
        backdoor = BackdoorPatch(single_pixel_backdoor=True, reverse_backdoor=True)
    elif attack_name == "random":
        backdoor = BackdoorPatch(pattern=pattern,
                                 alpha=RANDOM_BACKDOOR_ALPHA)
        aux_data = pattern
    elif attack_name in {"fixed", "reversed_fixed"}:
        backdoor = BackdoorPatch(pattern=pattern,
                                 alpha=FIXED_BACKDOOR_ALPHA)
    elif attack_name in {"sinusoid", "reversed_sinusoid"}:
        backdoor = BackdoorPatch(pattern=pattern,
                                 alpha=SINUSOID_BACKDOOR_ALPHA)
    elif attack_name == "warped":
        id_grid, noise_grid = None, None
        if aux_data:
            id_grid, noise_grid = aux_data
        backdoor = WarpingAttack(img_size[0], identity_grid=id_grid, noise_grid=noise_grid)
        aux_data = (backdoor.identity_grid, backdoor.noise_grid)
    else:
        raise NameError(f"Attack type {attack_name} is not a valid attack type.")
    backdoor_transform = transforms.Compose([backdoor, ClampRangeTransform()])
    return backdoor_transform, aux_data


def add_probe_data(probe_dict, key, indices, dataset, device, labels, transform=None,
                   track_idx=True, compute_diffs=False):
    if track_idx:
        probe_dict[f"{key}_idx"] = indices
    if transform is not None:
        if track_idx:
            probe_dict[f"{key}_original"] = torch.stack([dataset[i][0] for i in indices], dim=0).to(device)
        probe_dict[f"{key}"] = torch.stack([transform(dataset[i][0]) for i in indices], dim=0).to(device)
    else:
        probe_dict[f"{key}"] = torch.stack([dataset[i][0] for i in indices], dim=0).to(device)
    probe_dict[f"{key}_labels"] = torch.from_numpy(labels).to(device)
    if compute_diffs and transform is not None and track_idx:
        probe_dict[f"{key}_diff"] = probe_dict[f"{key}_original"] - probe_dict[f"{key}"]


def make_train_probes(num_classes, train_set, train_set_wo_aug, num_train_probes,
                      train_probe_attack, output_dir, main_proc, img_size, device):
    probes = {"backdoor": [], "clean": [], "backdoor_val": [], "clean_val": []}
    attack_target = np.random.choice(np.arange(num_classes))
    print("Chosen train probe target:", attack_target)
    train_indices = list(range(len(train_set)))
    probe_indices = np.random.choice(train_indices, size=(4 * num_train_probes), replace=False)
    probes["all_backdoor_idx"] = probe_indices

    base_idx, val_idx = probe_indices[:2*num_train_probes], probe_indices[2*num_train_probes:]
    # TODO: Add sleeper probe option...
    for suffix, indices in zip(('', '_val'), (base_idx, val_idx)):
        probe_transform, aux_data = make_probe_transform(train_probe_attack, img_size, output_dir, main_proc)
        backdoor_idx = indices[:num_train_probes]
        attack_labels = np.array([attack_target for i in backdoor_idx])
        add_probe_data(probes, "backdoor"+suffix, backdoor_idx, train_set_wo_aug, device, attack_labels,
                       transform=probe_transform, compute_diffs=True)

        clean_idx = indices[num_train_probes:]
        clean_labels = np.array([train_set_wo_aug[i][1] for i in clean_idx])

        add_probe_data(probes, "clean"+suffix, clean_idx, train_set, device, clean_labels)

    return probes, attack_target, aux_data


def make_val_probes(num_classes, dataset, train_set, train_set_wo_aug, num_val_probes, val_probe_attacks, output_dir,
                    main_proc, img_size, device, train_probe_indices):
    val_probes = {}
    attack_targets = {attack: np.random.choice(np.arange(num_classes)) for attack in val_probe_attacks}
    print("Chosen val attack targets:", attack_targets)
    attack_numbers = {attack: int(len(train_set) * num_val_probes[attack]) for attack in val_probe_attacks}
    print("Making attack image quantities:", attack_numbers)
    train_indices = list(range(len(train_set)))
    random_pattern = None
    warping_grids = None
    chosen_indices = np.array([], dtype=int)

    if 'sleeper' in val_probe_attacks:
        attack_idx = add_sleeper_probes(val_probes, dataset, 'train', train_set, chosen_indices,
                                        train_probe_indices, device, attack_numbers['sleeper'])
        chosen_indices = np.concatenate((chosen_indices, attack_idx))

    for attack in val_probe_attacks:
        if attack == 'sleeper':
            continue
        target = attack_targets[attack]
        num = attack_numbers[attack]

        # For clean attacks, get clean indices to choose from.
        indices_to_choose_from = train_indices if attack not in CLEAN_LABEL_ATTACKS else \
            np.where(train_set.targets == target)[0]

        # don't let multiple attacks hit the same image, including train probe images.
        indices_to_choose_from = [i for i in indices_to_choose_from if i not in np.concatenate((chosen_indices,
                                                                                                train_probe_indices))]

        attack_idx = np.random.choice(indices_to_choose_from, size=min(num, len(indices_to_choose_from)), replace=False)
        attack_labels = np.array([target for _ in attack_idx])
        probe_transform, aux_data = make_probe_transform(attack, img_size, output_dir, main_proc)
        add_probe_data(val_probes, f"backdoor_{attack}", attack_idx, train_set_wo_aug, device, attack_labels,
                       transform=probe_transform, compute_diffs=True)

        print(f"Backdoor ({attack}) probe shape:", val_probes[f"backdoor_{attack}"].shape)

        if attack == "random":
            random_pattern = aux_data
        if attack == "warped":
            warping_grids = aux_data

        # update chosen_indices:
        chosen_indices = np.concatenate((chosen_indices, attack_idx))

    val_probes["all_backdoor_idx"] = chosen_indices

    return val_probes, attack_targets, random_pattern, warping_grids


def make_test_probes(test_set, dataset, num_test_probes, val_probe_attacks, attack_targets, random_pattern, warping_grids,
                     output_dir, main_proc, img_size, device):
    test_probes = {}
    all_test_indices = list(range(len(test_set)))

    if 'sleeper' in val_probe_attacks:
        add_sleeper_probes(test_probes, dataset, 'test', test_set, np.array([]),
                           np.array([]), device, num_test_probes, track_idx=False)

    for attack in val_probe_attacks:
        if attack == 'sleeper':
            continue
        target = attack_targets[attack]

        aux_data = None
        if attack == "random":
            aux_data = random_pattern
        if attack == "warped":
            aux_data = warping_grids
        probe_transform, _ = make_probe_transform(attack, img_size, output_dir, main_proc, aux_data=aux_data)

        # Don't use any clean indices -- this way, the attack success rate should start at 0.
        #   (though practically there will be some randomly classified training images with low test acc.)
        non_clean_test_indices = np.where(test_set.targets != target)[0]
        test_indices = np.random.choice(non_clean_test_indices, size=min(len(non_clean_test_indices), num_test_probes),
                                        replace=False)
        test_labels = np.array([target for _ in test_indices])

        add_probe_data(test_probes, f"backdoor_{attack}", test_indices, test_set, device, test_labels,
                       transform=probe_transform, track_idx=False)

    return test_probes


sleeper_classes = {
    'cifar10': {
        'train': 5,  # Train images are dogs
        'test':  8
    }
}

DATA_ROOT = "./data/"  # TODO: set this better, somehow resolve circular import risk?


def add_sleeper_probes(probe_dict, dataset, split, train_set, chosen_indices, train_probe_indices, device,
                       num_idx, track_idx=True, compute_diffs=False):
    """
    probe_dict: dict to add data into
    dataset: string dataset name
    split: {'probe', 'train', 'test'}
    train_set: train set (or test set; this is poorly named)
    chosen_indices: indices that have already been chosen for other attacks, so avoid these. Should be empty...
    train_probe_indices: indices for the train probes, avoid these too.
    device: to load data onto.
    num_idx: how many images to generate.
    track_idx: save the indices?
    compute_diff: compute the diff images?
    """
    indices_to_choose_from = []
    attack_label = sleeper_classes[dataset]['train']  # Sleeper label is the same always.
    class_name = train_set.classes[sleeper_classes[dataset][split]].split()[-1]

    patch = Image.open(DATA_ROOT + f"sleeper_{dataset}/trigger_10.png")
    patch_size = 3
    patch = transforms.Resize(patch_size)(patch)  # TODO: is the size ever different?

    # TODO: probe?
    data_dir = DATA_ROOT + f"sleeper_{dataset}/{split}/{class_name}/"  # TODO: import data dir from dataset_utils?
    image_size = train_set.data[0].shape[0]

    indices_to_remove = []  # TODO: delete non-diff images to save time?
    sleeper_images = []
    indices_targeted = []
    for _, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.png'):
                img = Image.open(data_dir + file)
                image_index = int(file.split('.')[0])
                if split == 'train':
                    base_img = train_set.data[image_index]
                    if (img - base_img).sum() == 0:
                        indices_to_remove.append(image_index)
                    elif image_index in np.concatenate((chosen_indices, train_probe_indices)):
                        continue
                    else:
                        sleeper_images.append(img)
                        indices_targeted.append(image_index)
                if split == 'test':
                    loc = np.random.randint(0, image_size - (patch_size - 1), size=(2,))
                    img.paste(patch, box=tuple(loc))
                    sleeper_images.append(img)
                    indices_targeted.append(image_index)

    sleeper_images = torch.stack([transforms.ToTensor()(img) for img in sleeper_images])
    indices_targeted = np.array(indices_targeted)
    random_idx = np.random.choice(range(len(sleeper_images)), size=min(num_idx, len(sleeper_images)), replace=False)

    sleeper_images = sleeper_images[random_idx]
    indices_targeted = indices_targeted[random_idx]

    attack_labels = np.array([attack_label for _ in indices_targeted])

    if track_idx:
        probe_dict[f"backdoor_sleeper_idx"] = indices_targeted
        probe_dict[f"backdoor_sleeper_original"] = torch.stack([transforms.ToTensor()(train_set.data[i])
                                                                for i in indices_targeted], dim=0).to(device)
    probe_dict[f"backdoor_sleeper"] = sleeper_images.to(device)
    probe_dict[f"backdoor_sleeper_labels"] = torch.from_numpy(attack_labels).to(device)
    if compute_diffs and track_idx:
        probe_dict[f"backdoor_sleeper_diff"] = probe_dict[f"backdoor_sleeper_original"] - probe_dict[f"backdoor_sleeper"]
    return indices_targeted
