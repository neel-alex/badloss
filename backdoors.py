import os
import collections

import numpy as np
import torch
from torchvision import transforms
import cv2
from PIL import Image, ImageChops

import dist_utils


RANDOM_BACKDOOR_ALPHA = 0.075
FIXED_BACKDOOR_ALPHA = 0.025
SINUSOID_BACKDOOR_ALPHA = 0.075
SINUSOID_BACKDOOR_FREQ = 6
IMAGENETTE_ALPHA = 0.2

BLENDING_ATTACKS = {'random', 'fixed', 'sinusoid', 'warped', 'narcissus', 'frequency'}
BOOSTING_RATIO = 2
CLEAN_LABEL_ATTACKS = {'sinusoid', 'narcissus'}


class BackdoorPatch(object):
    def __init__(self, single_pixel_backdoor=False, reverse_backdoor=False,
                 pattern=None, alpha=None, mode='average', four_corners=False):
        assert pattern is None or (not single_pixel_backdoor and not reverse_backdoor and alpha is not None)
        print(f"BACKDOOR PATCH: {four_corners}")

        self.single_pixel_backdoor = single_pixel_backdoor
        self.reverse_backdoor = reverse_backdoor

        self.pattern = pattern
        self.alpha = alpha
        self.mode = mode

        self.four_corners = four_corners

    def __call__(self, tensor):
        backdoor_pix_val = 1.0

        if self.pattern is not None and self.mode == 'average':
            tensor = (1 - self.alpha) * tensor + (self.alpha) * self.pattern
        elif self.pattern is not None and self.mode == 'add':
            tensor = tensor + self.pattern * self.alpha
        elif self.single_pixel_backdoor:
            if self.reverse_backdoor:
                tensor[:, 1, 1] = backdoor_pix_val
            elif self.four_corners:
                tensor[:, 1, 1] = backdoor_pix_val
                tensor[:, 1, tensor.shape[2] - 2] = backdoor_pix_val
                tensor[:, tensor.shape[1] - 2, 1] = backdoor_pix_val
                tensor[:, tensor.shape[1] - 2, tensor.shape[2] - 2] = backdoor_pix_val
            else:
                tensor[:, tensor.shape[1] - 2, tensor.shape[2] - 2] = backdoor_pix_val
        else:
            if self.reverse_backdoor:
                tensor[:, 1, 1] = backdoor_pix_val
                tensor[:, 3, 1] = backdoor_pix_val
                tensor[:, 1, 3] = backdoor_pix_val
                tensor[:, 2, 2] = backdoor_pix_val
            elif self.four_corners:
                tensor[:, 1, 1] = backdoor_pix_val
                tensor[:, 3, 1] = backdoor_pix_val
                tensor[:, 1, 3] = backdoor_pix_val
                tensor[:, 2, 2] = backdoor_pix_val

                tensor[:, 1, tensor.shape[2] - 2] = backdoor_pix_val
                tensor[:, 3, tensor.shape[2] - 2] = backdoor_pix_val
                tensor[:, 1, tensor.shape[2] - 4] = backdoor_pix_val
                tensor[:, 2, tensor.shape[2] - 3] = backdoor_pix_val

                tensor[:, tensor.shape[1] - 2, 1] = backdoor_pix_val
                tensor[:, tensor.shape[1] - 4, 1] = backdoor_pix_val
                tensor[:, tensor.shape[1] - 2, 3] = backdoor_pix_val
                tensor[:, tensor.shape[1] - 3, 2] = backdoor_pix_val

                tensor[:, tensor.shape[1] - 2, tensor.shape[2] - 2] = backdoor_pix_val
                tensor[:, tensor.shape[1] - 4, tensor.shape[2] - 2] = backdoor_pix_val
                tensor[:, tensor.shape[1] - 2, tensor.shape[2] - 4] = backdoor_pix_val
                tensor[:, tensor.shape[1] - 3, tensor.shape[2] - 3] = backdoor_pix_val
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
    def __init__(self, img_size, s=0.5, k=4, grid_rescale=1.0, identity_grid=None, noise_grid=None, boost=1):
        self.img_size = img_size
        self.s = s * boost
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


class FrequencyAttack(object):
    def __init__(self, window_size=32, positions=((15, 15), (31, 31)), magnitude=30, boost=1):
        super().__init__()
        self.window_size = window_size
        self.positions = positions
        self.magnitude = magnitude * boost

    def __call__(self, x):
        import copy
        x_original = copy.deepcopy(x)
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
        if dataset == "mnist":
            freq = 1/4
        else:
            freq = SINUSOID_BACKDOOR_FREQ
        pattern = np.zeros(img_size, dtype=np.float32)
        if "reversed" in attack_name:
            for row in range(pattern.shape[0]):
                pattern[row, :, :] = 1 - np.cos(2 * np.pi * row * freq / pattern.shape[0])
        else:
            for col in range(pattern.shape[1]):
                pattern[:, col, :] = 1 - np.cos(2 * np.pi * col * freq / pattern.shape[1])
        pattern = transforms.ToTensor()(pattern)

    if "narcissus" in attack_name:
        pattern = np.load(f'data/{attack_name}_noise_{dataset}.npy')  # TODO: generate a new narcissus, currently just the same
        pattern = torch.tensor(pattern[0])

    return pattern


class Identity(object):
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        return x


def make_probe_transform(attack_name, img_size, dataset, output_dir, main_proc, aux_data=None, alpha_boost=1):
    if "random" in attack_name and aux_data is not None:
        pattern = aux_data
    else:
        pattern = get_pattern(attack_name, img_size, dataset, output_dir, main_proc)

    if attack_name == "patch":
        backdoor = BackdoorPatch(four_corners=(dataset == 'imagenette'))
    elif attack_name == "reversed_patch":
        backdoor = BackdoorPatch(reverse_backdoor=True)
    elif attack_name == "single_pix":
        backdoor = BackdoorPatch(single_pixel_backdoor=True, four_corners=(dataset == 'imagenette'))
    elif attack_name == "reversed_single_pix":
        backdoor = BackdoorPatch(single_pixel_backdoor=True, reverse_backdoor=True)
    elif attack_name == "random":
        backdoor = BackdoorPatch(pattern=pattern,
                                 alpha=RANDOM_BACKDOOR_ALPHA*alpha_boost)
        if dataset == "imagenette":
            backdoor = BackdoorPatch(pattern=pattern,
                                     alpha=IMAGENETTE_ALPHA*alpha_boost) # Try making stronger on imagenette

        aux_data = pattern
    elif attack_name in {"fixed", "reversed_fixed"}:
        backdoor = BackdoorPatch(pattern=pattern,
                                 alpha=FIXED_BACKDOOR_ALPHA*alpha_boost)
        if dataset == "imagenette":
            backdoor = BackdoorPatch(pattern=pattern,
                                     alpha=IMAGENETTE_ALPHA*alpha_boost) # Try making stronger on imagenette

    elif attack_name in {"sinusoid", "reversed_sinusoid"}:
        backdoor = BackdoorPatch(pattern=pattern,
                                 alpha=SINUSOID_BACKDOOR_ALPHA*alpha_boost)
        if dataset == "imagenette":
            backdoor = BackdoorPatch(pattern=pattern,
                                     alpha=IMAGENETTE_ALPHA*alpha_boost) # Try making stronger on imagenette
    elif attack_name == "warped":
        id_grid, noise_grid = None, None
        if aux_data:
            id_grid, noise_grid = aux_data

        if dataset == "gtsrb":
            backdoor = WarpingAttack(img_size[0], s=1.0, k=8, identity_grid=id_grid, noise_grid=noise_grid)
        else:
            backdoor = WarpingAttack(img_size[0], identity_grid=id_grid, noise_grid=noise_grid, boost=alpha_boost)

        aux_data = (backdoor.identity_grid, backdoor.noise_grid)
    elif attack_name == "narcissus" or attack_name == "alt_narcissus":
        backdoor = BackdoorPatch(pattern=pattern, alpha=1*alpha_boost, mode='add')
    elif attack_name in {"frequency", "reversed_frequency"}:
        backdoor = FrequencyAttack(boost=alpha_boost)

        if dataset == "imagenette":
            backdoor = FrequencyAttack(boost=alpha_boost, magnitude=120) # Try making stronger on imagenette
    elif 'clean' in attack_name:
        backdoor = Identity()
    else:
        raise NameError(f"Attack type {attack_name} is not a valid attack type.")
    backdoor_transform = transforms.Compose([backdoor, ClampRangeTransform()])
    return backdoor_transform, aux_data


def add_probe_data(probe_dict, key, indices, dataset, device, labels, transform=None,
                   track_idx=True, compute_diffs=False, override_dict_key=True):
    if not override_dict_key:
        assert key in probe_dict, "key should be in the data dict when concatenation is desired"
        assert transform is None, "Only supported for mislabeled probe"
        print(f"!! [WARNING] Key ({key}) already found in the probe dict. Appending new data to it.")
        if f"{key}_idx" in probe_dict:
            assert isinstance(probe_dict[f"{key}_idx"], np.ndarray)
            probe_dict[f"{key}_idx"] = np.concatenate([probe_dict[f"{key}_idx"], indices])

        # Concatenate the new examples with the old examples
        new_ex = torch.stack([dataset[i][0] for i in indices], dim=0)
        probe_dict[f"{key}"] = torch.cat([probe_dict[f"{key}"], new_ex], dim=0)
        new_labels = torch.from_numpy(labels)
        probe_dict[f"{key}_labels"] = torch.cat([probe_dict[f"{key}_labels"], new_labels], dim=0)
    else:
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


def make_train_probes(num_classes, dataset, train_set_wo_aug, num_train_probes,
                      train_probe_attack, output_dir, main_proc, img_size, device,
                      val_probe_indices, include_harder_backdoor_probes=False,
                      include_val_probe_examples=False):
    probes = {"backdoor": [], "clean": [], "backdoor_val": [], "clean_val": []}
    attack_target = np.random.choice(np.arange(num_classes))
    print("Chosen train probe target:", attack_target)
    train_indices = list(range(len(train_set_wo_aug)))
    valid_indices = [i for i in train_indices if i not in val_probe_indices]
    base_splits = 4 if include_val_probe_examples else 2
    num_examples = (base_splits * num_train_probes)
    if include_harder_backdoor_probes:
        hard_probe_size = num_train_probes
        num_examples += hard_probe_size
    probe_indices = np.random.choice(valid_indices, size=num_examples, replace=False)
    if train_probe_attack == "reversed_sinusoid":
        class_set = np.array(train_set_wo_aug.targets)[np.array(valid_indices)]
        class_counts = collections.Counter(class_set)
        while class_counts[attack_target] < 4 * num_train_probes:
            attack_target = np.random.choice(np.arange(num_classes))
        clean_indices = np.array(valid_indices)[class_set == attack_target]
        attack_split = 2
        clean_split = 3 if include_harder_backdoor_probes else 2
        chosen_attack_indices = np.random.choice(clean_indices, size=(attack_split * num_train_probes), replace=False)
        valid_indices = [i for i in valid_indices if i not in chosen_attack_indices]
        chosen_clean_indices = np.random.choice(valid_indices, size=(clean_split * num_train_probes), replace=False)
        probe_indices = np.concatenate([chosen_attack_indices[:num_train_probes], chosen_clean_indices[:num_train_probes],
                                        chosen_attack_indices[num_train_probes:], chosen_clean_indices[num_train_probes:]])
    probes["all_backdoor_idx"] = probe_indices

    base_idx, val_idx = probe_indices[:2*num_train_probes], probe_indices[2*num_train_probes:4*num_train_probes]
    # TODO: Add sleeper probe option...
    for suffix, indices in zip(('', '_val'), (base_idx, val_idx)):
        if not include_val_probe_examples and suffix == '_val':
            break
        probe_transform, aux_data = make_probe_transform(train_probe_attack, img_size, dataset, output_dir, main_proc)
        backdoor_idx = indices[:num_train_probes]
        if 'clean' in train_probe_attack:
            attack_labels = np.array([train_set_wo_aug[i][1] for i in backdoor_idx])
        else:
            attack_labels = np.array([attack_target for i in backdoor_idx])
        add_probe_data(probes, "backdoor"+suffix, backdoor_idx, train_set_wo_aug, device, attack_labels,
                       transform=probe_transform, compute_diffs=True)

        clean_idx = indices[num_train_probes:]
        clean_labels = np.array([train_set_wo_aug[i][1] for i in clean_idx])

        add_probe_data(probes, "clean"+suffix, clean_idx, train_set_wo_aug, device, clean_labels)

    if include_harder_backdoor_probes:
        mislabeled_probe_idx = probe_indices[base_splits*num_train_probes:]
        print(f"!! Adding {len(mislabeled_probe_idx)} harder backdoor examples with mislabeled probe...")
        orig_labels = np.array([train_set_wo_aug[i][1] for i in mislabeled_probe_idx])
        all_classes = np.arange(num_classes)
        probe_labels = np.array([np.random.choice(all_classes[orig_label != all_classes]) for orig_label in orig_labels])
        add_probe_data(probes, "backdoor", mislabeled_probe_idx, train_set_wo_aug, device, probe_labels,
                       override_dict_key=False)

    if not include_val_probe_examples:
        del probes["clean_val"]
        del probes["backdoor_val"]
        probes['all_backdoor_idx'] = np.concatenate((probes['backdoor_idx'], probes['clean_idx']))
        # clean_shape = probes["clean"].shape
        # probes["clean_val"] = torch.zeros((0, *clean_shape[1:]), dtype=probes["clean"].dtype)
        # probes["clean_val_labels"] = torch.zeros((0,), dtype=probes["clean_labels"].dtype)
        # probes["backdoor_val"] = probes["clean_val"].clone()
        # probes["backdoor_val_labels"] = torch.zeros((0,), dtype=probes["backdoor_labels"].dtype)
        # print(f"!! Setting shape for no val probes case / Clean shape: {clean_shape} / Validation shape: {probes['clean_val'].shape}")

    return probes, attack_target, aux_data


def make_val_probes(num_classes, dataset, train_set_wo_aug, num_val_probes, val_probe_attacks, output_dir,
                    main_proc, img_size, device):
    val_probes = {}
    attack_targets = {attack: np.random.choice(np.arange(num_classes)) for attack in val_probe_attacks}
    if 'sinusoid' in val_probe_attacks and dataset == "gtsrb":
        class_counts = collections.Counter(train_set_wo_aug.targets)
        while class_counts[attack_targets['sinusoid']] < 1000:
            attack_targets['sinusoid'] = np.random.choice(np.arange(num_classes))
    if 'sleeper' in val_probe_attacks:
        attack_targets['sleeper'] = sleeper_classes[dataset]['train']
    if 'narcissus' in val_probe_attacks:
        attack_targets['narcissus'] = narcissus_classes[dataset]
    print("Chosen val attack targets:", attack_targets)
    attack_numbers = {attack: int(len(train_set_wo_aug) * num_val_probes[attack]) for attack in val_probe_attacks}
    print("Making attack image quantities:", attack_numbers, "(clean label attacks may be incorrect)")
    train_indices = list(range(len(train_set_wo_aug)))
    random_pattern = None
    warping_grids = None
    chosen_indices = np.array([], dtype=int)

    if 'sleeper' in val_probe_attacks:
        attack_idx = add_sleeper_probes(val_probes, dataset, 'train', train_set_wo_aug, img_size,
                                        device, attack_numbers['sleeper'])
        chosen_indices = np.concatenate((chosen_indices, attack_idx))
        print(f"Backdoor (sleeper) probe shape:", val_probes[f"backdoor_sleeper"].shape)

    for attack in val_probe_attacks:
        if attack == 'sleeper':
            continue
        target = attack_targets[attack]
        num = attack_numbers[attack]
        # if attack == 'single_pix':
        #     num = 1
        # For clean attacks, get clean indices to choose from.
        indices_to_choose_from = train_indices
        if attack in CLEAN_LABEL_ATTACKS:
            indices_to_choose_from = np.where(np.array(train_set_wo_aug.targets) == target)[0]
            # Clean label attacks are expressed as a fraction of the target class! Adjust attack number appropriately.
            num = int(num_val_probes[attack] * len(indices_to_choose_from))
            if dataset == "gtsrb":
                # TODO: something more principled...
                num = max(num, 300)


        # don't let multiple attacks hit the same image, including train probe images.
        indices_to_choose_from = [i for i in indices_to_choose_from if i not in chosen_indices]

        attack_idx = np.random.choice(indices_to_choose_from, size=min(num, len(indices_to_choose_from)), replace=False)
        attack_labels = np.array([target for _ in attack_idx])
        probe_transform, aux_data = make_probe_transform(attack, img_size, dataset, output_dir, main_proc)
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
        add_sleeper_probes(test_probes, dataset, 'test', test_set, img_size, device, num_test_probes, track_idx=False)

    for attack in val_probe_attacks:
        if attack == 'sleeper':
            continue
        target = attack_targets[attack]

        aux_data = None
        if attack == "random":
            aux_data = random_pattern
        if attack == "warped":
            aux_data = warping_grids
        probe_transform, _ = make_probe_transform(attack, img_size, dataset, output_dir, main_proc, aux_data=aux_data)

        # Don't use any clean indices -- this way, the attack success rate should start at 0.
        #   (though practically there will be some randomly classified training images with low test acc.)
        non_clean_test_indices = np.where(np.array(test_set.targets) != target)[0]
        test_indices = np.random.choice(non_clean_test_indices, size=min(len(non_clean_test_indices), num_test_probes),
                                        replace=False)
        test_labels = np.array([target for _ in test_indices])

        add_probe_data(test_probes, f"backdoor_{attack}", test_indices, test_set, device, test_labels,
                       transform=probe_transform, track_idx=False)
        if attack in BLENDING_ATTACKS:
            boosted_transform, _ = make_probe_transform(attack, img_size, dataset, output_dir, main_proc,
                                                        aux_data=aux_data, alpha_boost=BOOSTING_RATIO)
            add_probe_data(test_probes, f"backdoor_{attack}_boosted", test_indices, test_set, device, test_labels,
                           transform=boosted_transform, track_idx=False)

    return test_probes


sleeper_classes = {
    'cifar10': {
        'train': 6,  # Train images are frogs
        'test':  4   # Source class is deer
    }
}

narcissus_classes = {
    'cifar10': 2
}

DATA_ROOT = "./data/"  # TODO: set this better, somehow resolve circular import risk?


def add_sleeper_probes(probe_dict, dataset, split, train_set_wo_aug, image_size, device,
                       num_idx, track_idx=True, compute_diffs=False):
    """
    probe_dict: dict to add data into
    dataset: string dataset name
    split: {'probe', 'train', 'test'}
    train_set_wo_aug: train set (or test set; this is poorly named)
    chosen_indices: indices that have already been chosen for other attacks, so avoid these. Should be empty...
    train_probe_indices: indices for the train probes, avoid these too.
    device: to load data onto.
    num_idx: how many images to generate.
    track_idx: save the indices?
    compute_diff: compute the diff images?
    """
    indices_to_choose_from = []
    attack_label = sleeper_classes[dataset]['train']  # Sleeper label is the same always.
    class_name = train_set_wo_aug.classes[sleeper_classes[dataset][split]].split()[-1]

    patch = Image.open(DATA_ROOT + f"sleeper_{dataset}/trigger_10.png")
    patch_size = 8
    patch = transforms.Resize(patch_size)(patch)  # TODO: is the size ever different?

    # TODO: probe?
    data_dir = DATA_ROOT + f"sleeper_{dataset}/{split}/{class_name}/"  # TODO: import data dir from dataset_utils?

    indices_to_remove = []  # TODO: delete non-diff images to save time?
    sleeper_images = []
    indices_targeted = []
    for _, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.png'):
                img = Image.open(data_dir + file)
                image_index = int(file.split('.')[0])
                if split == 'train':
                    base_img = train_set_wo_aug[image_index][0]
                    if ImageChops.difference(img, transforms.ToPILImage()(base_img)).getbbox() is None:
                        indices_to_remove.append(image_index)
                    else:
                        sleeper_images.append(img)
                        indices_targeted.append(image_index)
                if split == 'test':
                    loc = np.random.randint(0, image_size[0] - (patch_size - 1), size=(2,))
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
        probe_dict[f"backdoor_sleeper_original"] = torch.stack([train_set_wo_aug[i][0]
                                                                for i in indices_targeted], dim=0).to(device)
    probe_dict[f"backdoor_sleeper"] = sleeper_images.to(device)
    probe_dict[f"backdoor_sleeper_labels"] = torch.from_numpy(attack_labels).to(device)
    if compute_diffs and track_idx:
        probe_dict[f"backdoor_sleeper_diff"] = probe_dict[f"backdoor_sleeper_original"] - probe_dict[f"backdoor_sleeper"]
    return indices_targeted
