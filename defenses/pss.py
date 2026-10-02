"""Poisoned sample separation via feature consistency (PSS). Not evaluated in the paper.

Briefly train a model without augmentation, finetune it with an intra-class loss (pushing class centers apart),
then measure each example's feature consistency under random rotation/translation. Low-consistency examples are
treated as clean, high-consistency ones as poisoned; the attacked model is then alternately unlearned on the
poisoned and relearned on the clean split.
"""
import os

import torch
from torchvision import transforms
from tqdm import tqdm

from defenses.feature_extractor import FeatureExtractor
from torch_utils import train, test


def train_intraclass(backdoor_model, device, loader, optimizer, num_classes):
    """One epoch minimizing the mean cosine similarity between class feature centers."""
    backdoor_model.train()
    backdoor_model.to(device)
    backdoor_model_fe = FeatureExtractor(backdoor_model)

    for (data, target), ex_idx in tqdm(loader):
        data = data.to(device)
        target = target.to(device)
        outputs, features = backdoor_model_fe(data)

        centers = []
        for j in range(num_classes):
            j_idx = torch.where(target == j)[0]
            if j_idx.shape[0] == 0:
                continue
            centers.append(torch.mean(features[j_idx], dim=0))
        centers = torch.nn.functional.normalize(torch.stack(centers, dim=0), dim=1)
        similarity_matrix = torch.matmul(centers, centers.T)
        mask = torch.eye(similarity_matrix.shape[0], dtype=torch.bool).to(device)
        similarity_matrix[mask] = 0.0
        loss = torch.mean(similarity_matrix)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    backdoor_model_fe.remove()


def calc_fct(backdoor_model, device, loader):
    """Per-example feature consistency: squared feature change under a random rotation + translation."""
    backdoor_model.eval()
    backdoor_model.to(device)
    backdoor_model_fe = FeatureExtractor(backdoor_model)
    fct_transform = transforms.Compose([
        transforms.RandomRotation(180),
        transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),
    ])

    fcts = torch.zeros((len(loader.dataset),)).to(device)
    for (data, target), ex_idx in tqdm(loader):
        data = data.to(device)
        data2 = fct_transform(data)
        with torch.no_grad():
            outputs1, features1 = backdoor_model_fe(data)
            outputs2, features2 = backdoor_model_fe(data2)
        fcts[ex_idx] = torch.mean((features1 - features2)**2, dim=1)
    backdoor_model_fe.remove()
    return fcts


def pss_unlearn(model, device, clean_dl, pois_dl, optimizer, criterion):
    """One epoch of gradient ascent on the poisoned split followed by one of descent on the clean split."""
    model.train()
    model.to(device)
    for sign, loader in ((-1, pois_dl), (1, clean_dl)):
        pbar = tqdm(loader)
        for (data, target), ex_idx in pbar:
            data = data.to(device)
            target = target.to(device)
            loss = criterion(model(data), target).mean()  # Reduction has been disabled -- do explicit reduction
            if sign < 0:
                loss = -loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_description(f"Loss: {float(loss.detach()):.4f}")


def run(exp):
    args = exp.args
    exp.pretrain()
    exp.report_attacked_model()
    exp.wandb_prefix = "retraining_"

    # Step 1: briefly train a fresh model without augmentation
    backdoor_model = exp.new_model()
    optimizer = torch.optim.SGD(backdoor_model.parameters(), lr=args.pss_lr, momentum=0.9, weight_decay=5e-4)

    print("!! Performing PSS initial pretraining...")
    output_checkpoint_file = os.path.join(exp.output_dir, "pss_model_pretrain.pth")
    criterion = exp.criterion
    if not os.path.exists(output_checkpoint_file):
        criterion = torch.nn.CrossEntropyLoss(reduction='none').to(exp.device)
        for epoch in tqdm(range(args.pss_pretrain_epochs)):
            train(backdoor_model, exp.device, exp.new_idx_loader_wo_aug, optimizer, criterion)
            if epoch % 2 == 1:
                exp.evaluate(backdoor_model, criterion)
        torch.save(backdoor_model.state_dict(), output_checkpoint_file)
    else:
        print("!! Loading pretrained checkpoint file:", output_checkpoint_file)
        backdoor_model.load_state_dict(torch.load(output_checkpoint_file, map_location=exp.device))

    # Step 2: intra-class finetuning without augmentation
    optimizer = torch.optim.SGD(backdoor_model.parameters(), lr=args.pss_lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    output_checkpoint_file = os.path.join(exp.output_dir, "pss_model_intraclass.pth")
    print("!! Performing intraclass training...")
    if not os.path.exists(output_checkpoint_file):
        for epoch in tqdm(range(args.pss_intraclass_epochs)):
            train_intraclass(backdoor_model, exp.device, exp.new_idx_loader_wo_aug, optimizer, exp.num_classes)
            scheduler.step()
            if epoch % 5 == 4:
                exp.evaluate(backdoor_model, criterion)
        torch.save(backdoor_model.state_dict(), output_checkpoint_file)
    else:
        print("!! Loading intraclass trained checkpoint file:", output_checkpoint_file)
        backdoor_model.load_state_dict(torch.load(output_checkpoint_file, map_location=exp.device))

    # Step 3: split the training set by feature consistency
    print("!! Calculating FCT metric...")
    fcts = calc_fct(backdoor_model, exp.device, exp.new_idx_loader_wo_aug).cpu()
    sorted_fcts = fcts[fcts.nonzero()[:, 0]].sort()[0]  # Zeros: indices not in the training set
    lower_limit = sorted_fcts[int(len(sorted_fcts) * args.pss_clean_quantile)].item()
    upper_limit = sorted_fcts[int(len(sorted_fcts) * args.pss_poison_quantile)].item()
    clean_dl = exp.loader(torch.where((fcts > 0) & (fcts < lower_limit))[0], dataset=exp.new_idx_loader.dataset)
    pois_dl = exp.loader(torch.where(fcts >= upper_limit)[0], dataset=exp.new_idx_loader.dataset)

    # Step 4: alternately unlearn the poisoned split and relearn the clean split, starting from the attacked model
    model = exp.model
    optimizer = torch.optim.SGD(model.parameters(), lr=args.pss_unlearn_lr, momentum=0.9, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss(reduction='none').to(exp.device)
    print("!! Performing PSS backdoor defense...")
    output_checkpoint_file = os.path.join(exp.output_dir, "pss_model.pth")
    if not os.path.exists(output_checkpoint_file):
        for epoch in tqdm(range(args.pss_unlearn_epochs)):
            pss_unlearn(model, exp.device, clean_dl, pois_dl, optimizer, criterion)
            if epoch % 5 == 4:
                exp.evaluate(model, criterion)
        torch.save(model.state_dict(), output_checkpoint_file)
    else:
        print("!! Loading PSS-trained checkpoint file:", output_checkpoint_file)
        model.load_state_dict(torch.load(output_checkpoint_file, map_location=exp.device))

    print("Retrained model performance:")
    test_stats, _ = test(model, exp.device, criterion, exp.test_idx_loader, exp.distributed, exp.rank)
    exp.record_model_metrics("retrained", test_stats, exp.evaluate_asr(model, criterion))
    print("Done with PSS")
