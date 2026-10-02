"""Causality-inspired Backdoor Defense (Zhang et al., 2023). Not evaluated in the paper.

Briefly train a "backdoored" model (which learns the easy, backdoor-like features), then train a clean model
whose features are pushed to be independent of the backdoored model's (WGAN-style mutual-information
minimization), with examples reweighted towards those the backdoored model finds hard.

The discriminator code is from https://github.com/zaixizhang/CBD/blob/main/utils/util.py.
"""
import os

import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from defenses.feature_extractor import FeatureExtractor
from torch_utils import train


def shuffle(real):
    """Shift a batch by one (P(X,Y) -> P(X)P(Y)): [1, 2, 3, 4, 5] -> [2, 3, 4, 5, 1]."""
    batch_size = real.size(0)
    shuffled_index = ((torch.arange(batch_size) + 1) % batch_size).to(real.device)
    return real.index_select(dim=0, index=shuffled_index)


def spectral_norm(W, n_iteration=5):
    """Spectral norm of a weight matrix (or bias vector) by power iteration."""
    if W.dim() == 1:
        W = W.unsqueeze(-1)
    out_dim, in_dim = W.size()
    Wt = W.transpose(0, 1)
    u = torch.ones(1, in_dim).to(W.device)
    for _ in range(n_iteration):
        v = torch.mm(u, Wt)
        v = v / v.norm(p=2)
        u = torch.mm(v, W)
        u = u / u.norm(p=2)
    return torch.mm(torch.mm(u, Wt), v.transpose(0, 1)).sum() ** 0.5


class MLP(nn.Module):
    def __init__(self, in_dim, n_classes, hidden_dim, dropout, n_layers=2, act=F.leaky_relu):
        super().__init__()
        self.l_in = nn.Linear(in_dim, hidden_dim)
        self.l_hs = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers - 2))
        self.l_out = nn.Linear(hidden_dim, n_classes)
        self.dropout = nn.Dropout(p=dropout)
        self.act = act

    def forward(self, input):
        hidden = self.act(self.l_in(self.dropout(input)))
        for l_h in self.l_hs:
            hidden = self.act(l_h(self.dropout(hidden)))
        return self.l_out(self.dropout(hidden))


class Disc(nn.Module):
    """2-layer discriminator for the mutual information estimate."""
    def __init__(self, x_dim, y_dim, dropout):
        super().__init__()
        self.disc = MLP(x_dim + y_dim, 1, y_dim, dropout, n_layers=2)

    def forward(self, x, y):
        return self.disc(torch.cat((x, y), dim=-1)).squeeze(-1)


class DisenEstimator(nn.Module):
    """MI(X, Y) = E_pxy[T(x, y)] - E_pxpy[T(x, y)], estimated adversarially with a spectrally normalized critic."""
    def __init__(self, dim1, dim2, dropout):
        super().__init__()
        self.disc = Disc(dim1, dim2, dropout)

    def forward(self, x, y):
        sy = shuffle(y)
        loss = self.disc(x, y).mean() - self.disc(x, sy).mean()
        return loss*0.01

    def spectral_norm(self):
        """Lipschitz constraint for the WGAN critic."""
        with torch.no_grad():
            for w in self.parameters():
                w.data /= spectral_norm(w.data)


def train_cbd(clean_model, backdoor_model, discriminator, device, loader, optimizer, adv_optimizer, criterion,
              ce_gamma=1.0):
    """One epoch: a pass maximizing the discriminator's MI estimate, then a pass training the clean model."""
    backdoor_model.eval()
    backdoor_model.to(device)
    clean_model.train()
    clean_model.to(device)
    discriminator.train()
    discriminator.to(device)

    backdoor_model_fe, clean_model_fe = FeatureExtractor(backdoor_model), FeatureExtractor(clean_model)

    pbar = tqdm(loader)
    for (data, target), ex_idx in pbar:
        data = data.to(device)
        output1, z_hidden = clean_model_fe(data)
        with torch.no_grad():
            output2, r_hidden = backdoor_model_fe(data)

        r_hidden, z_hidden = r_hidden.detach(), z_hidden.detach()
        dis_loss = - discriminator(r_hidden, z_hidden)
        adv_optimizer.zero_grad()
        dis_loss.backward()
        adv_optimizer.step()
        discriminator.spectral_norm()
        pbar.set_description(f"Loss: {float(dis_loss.detach()):.4f}")

    pbar = tqdm(loader)
    for (data, target), ex_idx in pbar:
        data = data.to(device)
        target = target.to(device)

        output1, z_hidden = clean_model_fe(data)
        with torch.no_grad():
            output2, r_hidden = backdoor_model_fe(data)
            loss_bias = criterion(output2, target)
            loss_d = criterion(output1, target).detach()

        r_hidden = r_hidden.detach()
        dis_loss = discriminator(r_hidden, z_hidden)

        # Upweight examples the backdoored model finds hard
        weight = (loss_bias / (loss_d + loss_bias + 1e-8)) ** ce_gamma
        weight = weight * weight.shape[0] / torch.sum(weight)
        loss = torch.mean(weight * criterion(output1, target))
        loss += dis_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pbar.set_description(f"Loss: {float(loss.detach()):.4f}")

    backdoor_model_fe.remove()
    clean_model_fe.remove()


def run(exp):
    args = exp.args
    exp.wandb_prefix = "retraining_"
    backdoor_model = exp.new_model()

    print("!! Performing CBD initial pretraining...")
    output_checkpoint_file = os.path.join(exp.output_dir, "cbd_model_pretrain.pth")
    criterion = exp.criterion
    if not os.path.exists(output_checkpoint_file):
        criterion, optimizer, _ = exp.new_optimizer(backdoor_model, args.cbd_pretrain_epochs)
        for epoch in tqdm(range(args.cbd_pretrain_epochs)):
            train(backdoor_model, exp.device, exp.new_idx_loader, optimizer, criterion)
            if epoch % 5 == 4:
                exp.evaluate(backdoor_model, criterion)
        torch.save(backdoor_model.state_dict(), output_checkpoint_file)
    else:
        print("!! Loading pretrained checkpoint file:", output_checkpoint_file)
        backdoor_model.load_state_dict(torch.load(output_checkpoint_file, map_location=exp.device))

    clean_model = exp.new_model()
    feature_dim = clean_model.fc.in_features  # ResNet penultimate (avgpool) features
    discriminator = DisenEstimator(feature_dim, feature_dim, dropout=0.2)
    adv_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.cbd_adv_lr)
    adv_scheduler = torch.optim.lr_scheduler.StepLR(adv_optimizer, step_size=20, gamma=0.1)
    optimizer = torch.optim.SGD(clean_model.parameters(), lr=args.cbd_lr, momentum=0.9, weight_decay=1e-4,
                                nesterov=True)

    output_checkpoint_file = os.path.join(exp.output_dir, "cbd_model.pth")
    if not os.path.exists(output_checkpoint_file):
        criterion = torch.nn.CrossEntropyLoss(reduction='none').to(exp.device)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 70], gamma=0.1)
        for epoch in range(exp.num_epochs):
            train_cbd(clean_model, backdoor_model, discriminator, exp.device, exp.new_idx_loader, optimizer,
                      adv_optimizer, criterion, args.cbd_ce_gamma)
            if epoch % 5 == 4:
                print(f"Evaluation at epoch {epoch+1}")
                exp.evaluate(clean_model, criterion)
            if epoch % 50 == 49:
                exp.evaluate_asr(clean_model, criterion)
            scheduler.step()
            adv_scheduler.step()
        torch.save(clean_model.state_dict(), output_checkpoint_file)
    else:
        print("!! Loading pretrained checkpoint file:", output_checkpoint_file)
        clean_model.load_state_dict(torch.load(output_checkpoint_file, map_location=exp.device))

    print("Retrained model performance:")
    test_stats = exp.evaluate(clean_model, criterion)
    exp.record_model_metrics("retrained", test_stats, exp.evaluate_asr(clean_model, criterion))
