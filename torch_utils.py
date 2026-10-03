import numpy as np
import torch
from torchvision import models
from tqdm import tqdm
import dist_utils
from dist_utils import DistributedSamplerWrapper


def get_model(dataset, num_classes, device, local_rank, verbose=False, arch='resnet50'):
    if arch in {'resnet18', 'resnet34', 'resnet50'}:
        model = getattr(models, arch)(weights=None, num_classes=num_classes)
        if dataset == "cifar10":  # Small-image stem: 3x3 conv, stride 1 (and a fresh classifier head)
            model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif arch == 'vgg16':
        model = models.vgg16_bn(weights=None, num_classes=num_classes)
    elif arch == 'densenet':
        model = models.densenet121(weights=None, num_classes=num_classes)
    elif arch == 'squeezenet':
        model = models.squeezenet1_0(weights=None, num_classes=num_classes)
    elif arch == 'efficientnet':
        model = models.efficientnet_b7(weights=None, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    model = model.to(device)
    if verbose:
        print(model)
    model = dist_utils.convert_to_distributed(model, local_rank=local_rank, sync_bn=True)
    return model


def get_optimizer(model, device, lr, wd, num_epochs):
    """AdamW with a cosine schedule over num_epochs; per-example (reduction='none') cross-entropy."""
    criterion = torch.nn.CrossEntropyLoss(reduction='none').to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    return criterion, optimizer, lr_scheduler


def train(model, device, train_loader, optimizer, criterion, log_interval=10, flooding_threshold=None):
    """One epoch of training. criterion must use reduction='none'. flooding_threshold enables loss flooding (ABL)."""
    model.train()
    optimizer.zero_grad()

    pbar = tqdm(train_loader)
    for batch_idx, ((data, target), ex_idx) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(data)
        loss = criterion(output, target)
        loss = torch.clamp(loss, max=100)
        if flooding_threshold is not None:
            loss = (loss - flooding_threshold).abs() + flooding_threshold

        assert loss.shape == (len(data),)
        loss = loss.mean()  # Reduction has been disabled -- do explicit reduction

        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            pbar.set_description(f"Loss: {float(loss.detach()):.4f}")
    pbar.close()


def test(model, device, criterion, test_loader, distributed, rank, set_name="Test", log_predictions=False):
    """Evaluate in eval mode. With log_predictions, also returns per-example losses/predictions."""
    model.eval()

    correct = torch.tensor([0]).to(device)
    test_loss = torch.tensor([0.0]).to(device)
    total = torch.tensor([0]).to(device)
    test_acc = 0.

    example_idx = []
    predictions = []
    targets = []
    loss_values = []

    for (data, target), ex_idx in test_loader:
        with torch.no_grad():
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss_vals = criterion(output, target)

            test_loss += float(loss_vals.sum())
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += len(data)

            if log_predictions:
                loss_values.append(loss_vals.detach().clone())
                predictions.append(output.argmax(dim=1).detach())
                example_idx.append(ex_idx.clone())
                targets.append(target.clone())

    # Reduce all of the values in case of distributed processing
    torch.cuda.synchronize()
    correct = int(dist_utils.reduce_tensor(correct.data))
    test_loss = float(dist_utils.reduce_tensor(test_loss.data))
    total = int(dist_utils.reduce_tensor(total.data))

    if isinstance(test_loader.sampler, DistributedSamplerWrapper):
        num_dataset_ex = len(test_loader.sampler.sampler)
    else:
        num_dataset_ex = len(test_loader.sampler)

    if not distributed:
        assert total == num_dataset_ex, f"{total} != {num_dataset_ex}"
    if total != num_dataset_ex:
        print(f"!! Warning -- aggregated total value ({total}) is not equal to the dataset size: {num_dataset_ex}...")
    if total > 0:
        test_loss /= total
        test_acc = 100. * correct / total
    output_dict = dict(loss=test_loss, acc=test_acc, correct=correct, total=total)
    if distributed:
        set_name = f"Rank: {rank} | {set_name}"
    print(f"{set_name} set | Average loss: {test_loss:.4f} | Accuracy: {correct}/{total} ({test_acc:.2f}%)")

    pred_output_dict = None
    if log_predictions:
        # Collect the statistics from all the GPUs
        example_idx = torch.cat(dist_utils.gather_tensor(torch.cat(example_idx, dim=0)), dim=0).detach().cpu().numpy()
        predictions = torch.cat(dist_utils.gather_tensor(torch.cat(predictions, dim=0)), dim=0).detach().cpu().numpy()
        targets = torch.cat(dist_utils.gather_tensor(torch.cat(targets, dim=0)), dim=0).detach().cpu().numpy()
        loss_values = torch.cat(dist_utils.gather_tensor(torch.cat(loss_values, dim=0)), dim=0).detach().cpu().numpy()
        pred_output_dict = {"ex_idx": example_idx, "preds": predictions, "targets": targets, "loss": loss_values}
    return output_dict, pred_output_dict


def test_tensor(model, device, criterion, data, target, msg=None, batch_size=None):
    """Evaluate (eval mode) on an in-memory tensor dataset; returns a stats dict."""
    assert torch.is_tensor(data) and torch.is_tensor(target)
    if len(data) == 0:
        return {}
    model.eval()
    data = data.to(device)
    target = target.to(device)
    with torch.no_grad():
        if batch_size is None:
            output = model(data)
            loss_vals = criterion(output, target)
            test_loss = float(loss_vals.mean())

            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct = pred.eq(target.view_as(pred)).sum().item()
            total = len(data)
        else:
            total = 0
            correct = 0
            loss_vals_list = []

            num_batches = int(np.ceil(len(data) / float(batch_size)))
            for i in range(num_batches):
                start, end = i * batch_size, (i+1) * batch_size
                output = model(data[start:end])
                loss_vals = criterion(output, target[start:end])
                loss_vals_list.append(loss_vals.detach())

                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target[start:end].view_as(pred)).sum().item()
                total += len(output)

            loss_vals = torch.cat(loss_vals_list, dim=0)
            test_loss = float(loss_vals.mean())

    test_acc = 100. * correct / total
    output_dict = dict(loss=test_loss, acc=test_acc, correct=correct, total=total)

    loss_vals = loss_vals.detach().cpu().numpy()
    output_dict["loss_mean"] = np.mean(loss_vals)
    output_dict["loss_var"] = np.var(loss_vals)
    output_dict["loss_std"] = np.std(loss_vals)

    header = "Test set" if msg is None else msg
    print(f"{header} | Loss mean: {output_dict['loss_mean']:.4f} | Loss std: {output_dict['loss_std']:.4f} | Accuracy: {test_acc:.2f}% ({correct}/{total})")

    return output_dict


def collect_losses(model, device, new_idx_loader_wo_aug, criterion):
    """Per-example loss and correct-class probability (eval mode) over a loader, indexed by example."""
    model.eval()
    loss_array = torch.zeros(len(new_idx_loader_wo_aug.dataset))
    probs_array = torch.zeros(len(new_idx_loader_wo_aug.dataset))

    for (data, target), ex_idx in new_idx_loader_wo_aug:
        with torch.no_grad():
            data, target = data.to(device), target.to(device)
            output = model(data)
            probs = torch.nn.functional.softmax(output, dim=1)
            correct_class_probs = probs[torch.arange(probs.shape[0]), target]
            loss_vals = criterion(output, target)

            loss_array[ex_idx] = loss_vals.detach().clone().cpu()
            probs_array[ex_idx] = correct_class_probs.detach().clone().cpu()

    return loss_array, probs_array
