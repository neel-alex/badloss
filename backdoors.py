import os

import numpy as np
import torch
from torchvision import transforms
import cv2

import dist_utils


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
    def __init__(self, img_size, s=0.5, k=4, grid_rescale=1.0):
        self.img_size = img_size
        self.s = s
        self.k = k
        self.grid_rescale = grid_rescale
        self.identity_grid = self.create_identity_grid(img_size)
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


def make_probes(num_classes, train_set, train_set_wo_aug, num_example_probes, attack_types,
                random_backdoor_alpha, output_dir, main_proc, img_size, device):
    probes = {"backdoor": [], "clean": []}

    chosen_attack_targets = {k: np.random.choice(np.arange(num_classes)) for k in attack_types}
    print("Chosen attack targets:", chosen_attack_targets)
    remaining_indices = list(range(len(train_set)))  # [i for i in range(len(train_set)) if i not in probes["backdoor_idx"]]
    new_indices = np.random.choice(remaining_indices, size=num_example_probes * len(attack_types), replace=False)
    probes.update({"all_backdoor_idx": new_indices})

    # Create a main random pattern
    pattern_file = os.path.join(output_dir, "random_pattern.png")
    if main_proc:
        random_pattern = np.clip(np.random.rand(*img_size) * 255, 0, 255)
        print("Random pattern shape:", random_pattern.shape)
        cv2.imwrite(pattern_file, random_pattern)
    dist_utils.wait_for_other_procs()  # Distributed barrier

    # Load the pattern to ensure the same pattern is loaded by all processes
    random_pattern_img = cv2.imread(pattern_file, cv2.IMREAD_UNCHANGED)
    random_pattern = transforms.ToTensor()(random_pattern_img)
    print(f"Random Pattern / Loaded shape: {random_pattern_img.shape} / Tensor shape: {random_pattern.shape} / Min: {random_pattern.min()} / Max: {random_pattern.max()}")

    for i, attack_type in enumerate(attack_types):
        if attack_type == "":
            backdoor = BackdoorPatch()
        elif attack_type == "random":
            backdoor = BackdoorPatch(pattern=random_pattern, alpha=random_backdoor_alpha)
        elif attack_type == "reversed":
            backdoor = BackdoorPatch(reverse_backdoor=True)
        elif attack_type == "single_pix":
            backdoor = BackdoorPatch(single_pixel_backdoor=True)
        elif attack_type == "reversed_single_pix":
            backdoor = BackdoorPatch(single_pixel_backdoor=True, reverse_backdoor=True)
        else:
            assert attack_type == "warped"
            backdoor = WarpingAttack(img_size[0])
        backdoor_transform = transforms.Compose([backdoor, ClampRangeTransform()])

        fake_backdoor_label = chosen_attack_targets[attack_type]
        print(f"!! Random class picked for {attack_type.replace('_', ' ')} backdoor: {fake_backdoor_label}")

        current_idx = probes["all_backdoor_idx"][i * num_example_probes:(i + 1) * num_example_probes]
        if attack_type == "":
            prefix = f"backdoor"
        else:
            prefix = f"backdoor_{attack_type}"

        probes[f"{prefix}_idx"] = current_idx
        probes[f"{prefix}_original"] = torch.stack([train_set_wo_aug[i][0] for i in current_idx], dim=0).to(device)
        probes[f"{prefix}"] = torch.stack([backdoor_transform(train_set_wo_aug[i][0]) for i in current_idx], dim=0).to(
            device)
        probes[f"{prefix}_diff"] = probes[f"{prefix}_original"] - probes[f"{prefix}"]
        probes[f"{prefix}_labels"] = torch.from_numpy(np.array([fake_backdoor_label for i in current_idx])).to(device)
        print(f"Backdoor ({attack_type}) probe shape:", probes["backdoor"].shape)

    remaining_indices = [i for i in range(len(train_set)) if i not in probes["all_backdoor_idx"]]
    new_indices = np.random.choice(remaining_indices, size=num_example_probes, replace=False)
    probes.update({"clean_idx": new_indices})
    probes["clean"] = torch.stack([train_set[i][0] for i in probes["clean_idx"]], dim=0).to(device)
    probes["clean_labels"] = torch.from_numpy(np.array([train_set_wo_aug[i][1] for i in probes["clean_idx"]])).to(
        device)
    print("Clean probe shape:", probes["clean"].shape)
    # In[ ]:
    return probes, chosen_attack_targets, random_pattern

