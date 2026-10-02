# CognitiveDistillation module taken directly from https://github.com/HanxunH/CognitiveDistillation

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


def total_variation_loss(img, weight=1):
    b, c, h, w = img.size()
    tv_h = torch.pow(img[:, :, 1:, :]-img[:, :, :-1, :], 2).sum(dim=[1, 2, 3])
    tv_w = torch.pow(img[:, :, :, 1:]-img[:, :, :, :-1], 2).sum(dim=[1, 2, 3])
    return weight*(tv_h+tv_w)/(c*h*w)


class CognitiveDistillation(nn.Module):
    def __init__(self, lr=0.1, p=1, gamma=0.01, beta=1.0, num_steps=100, mask_channel=1, norm_only=False):
        super(CognitiveDistillation, self).__init__()
        self.p = p
        self.gamma = gamma
        self.beta = beta
        self.num_steps = num_steps
        self.l1 = torch.nn.L1Loss(reduction='none')
        self.lr = lr
        self.mask_channel = mask_channel
        self.get_features = False
        self._EPSILON = 1.e-6
        self.norm_only = norm_only

    def get_raw_mask(self, mask):
        mask = (torch.tanh(mask) + 1) / 2
        return mask

    def forward(self, model, images, labels=None):
        model.eval()
        b, c, h, w = images.shape
        mask = torch.ones(b, self.mask_channel, h, w).to(images.device)
        mask_param = nn.Parameter(mask)
        optimizerR = torch.optim.Adam([mask_param], lr=self.lr, betas=(0.1, 0.1))
        if self.get_features:
            features, logits = model(images)
        else:
            logits = model(images).detach()
        for step in range(self.num_steps):
            optimizerR.zero_grad()
            mask = self.get_raw_mask(mask_param).to(images.device)
            x_adv = images * mask + (1-mask) * torch.rand(b, c, 1, 1).to(images.device)
            if self.get_features:
                adv_fe, adv_logits = model(x_adv)
                if len(adv_fe[-2].shape) == 4:
                    loss = self.l1(adv_fe[-2], features[-2].detach()).mean(dim=[1, 2, 3])
                else:
                    loss = self.l1(adv_fe[-2], features[-2].detach()).mean(dim=1)
            else:
                adv_logits = model(x_adv)
                loss = self.l1(adv_logits, logits).mean(dim=1)
            norm = torch.norm(mask, p=self.p, dim=[1, 2, 3])
            norm = norm * self.gamma
            loss_total = loss + norm + self.beta * total_variation_loss(mask)
            loss_total.mean().backward()
            optimizerR.step()
        mask = self.get_raw_mask(mask_param).detach().cpu()
        if self.norm_only:
            return torch.norm(mask, p=1, dim=[1, 2, 3])
        return mask.detach()


def run(exp):
    """Cognitive Distillation (Huang et al., 2023), adapted to filter the training set and retrain: learn a
    minimal input mask per example that preserves the model's output; remove the examples with the smallest
    masks (backdoor triggers are small and sufficient)."""
    args = exp.args
    exp.pretrain()
    exp.report_attacked_model()
    exp.wandb_prefix = "retraining_"

    cd = CognitiveDistillation(num_steps=args.cd_num_steps)
    masks = torch.zeros(len(exp.new_idx_loader_wo_aug.dataset), *exp.img_size[:-1])
    for (data, target), ex_idx in tqdm(exp.new_idx_loader_wo_aug):
        masks[ex_idx] = cd(exp.model, data.to(exp.device)).squeeze()

    mask_norms = torch.norm(masks, dim=(1, 2), p=1)
    valid_mask_idx = torch.where(mask_norms != 0)[0]
    base_idx = np.intersect1d(exp.train_probe['clean_idx'], valid_mask_idx.numpy())
    print("Num training examples for cognitive distillation", len(base_idx))

    # Standardize by the clean probes' statistics (only needed for CD's own thresholded detection)
    mean, std = mask_norms[base_idx].mean(), mask_norms[base_idx].std()
    mask_norms[valid_mask_idx] -= mean
    mask_norms[valid_mask_idx] /= std

    identified_indices = mask_norms.argsort()[:int(args.cd_remove_frac * len(mask_norms))]
    exp.report_detection(identified_indices)
    exp.retrain(identified_indices, checkpoint_tag=args.cd_remove_frac)
