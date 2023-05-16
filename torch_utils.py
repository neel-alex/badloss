from collections import OrderedDict

import numpy as np
import torch
from torchvision import models
from tqdm import tqdm
from catalyst.data import DistributedSamplerWrapper

import dist_utils


def get_model(dataset, num_classes, device, local_rank, verbose=False):
    if dataset == "mnist":
        # Create BadNet architecture (https://arxiv.org/abs/1708.06733)
        model = torch.nn.Sequential(OrderedDict([
            ('conv1', torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0)),
            ('act1', torch.nn.ReLU(inplace=True)),
            ('pool1', torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)),
            ('conv2', torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0)),
            ('act2', torch.nn.ReLU(inplace=True)),
            ('pool2', torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)),
            ('flatten', torch.nn.Flatten()),
            ('fc1', torch.nn.Linear(in_features=32 * 4 * 4, out_features=512)),
            ('fc1_act', torch.nn.ReLU(inplace=True)),
            ('fc', torch.nn.Linear(in_features=512, out_features=10)),
        ]))
        model = model.to(device)
    else:
        # Create ResNet-50
        model = models.resnet50(pretrained=False, num_classes=num_classes)
        if "cifar" in dataset:  # Change the first and last layer for cifar10/cifar100
            model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        model = model.to(device)
    if verbose:
        print(model)
    model = dist_utils.convert_to_distributed(model, local_rank=local_rank, sync_bn=True)
    return model


def get_optimizer(model, device, lr, momentum, wd, num_epochs):    
    criterion = torch.nn.CrossEntropyLoss(reduction='none').to(device)  # reduction='mean' by default
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler = torch.cuda.amp.GradScaler()
    return criterion, optimizer, lr_scheduler, scaler


def train(model, device, train_loader, optimizer, criterion, scaler, log_interval=10, log_predictions=False,
          use_autocast=False, flooding_threshold=None, loss_max_indices=None, gradient_ascent=False,
          flooding_type='flooding'):
    assert flooding_type in ['lga', 'flooding']
    model.train()
    optimizer.zero_grad()

    example_idx = []
    predictions = []
    targets = []
    loss_values = []

    pbar = tqdm(train_loader)
    for batch_idx, ((data, target), ex_idx) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=use_autocast):
            output = model(data)
            loss = criterion(output, target)
            loss = torch.clamp(loss, max=100)
            if flooding_threshold is not None:
                if flooding_type == 'lga':
                    loss = torch.sign(loss - flooding_threshold) * loss
                else:
                    assert flooding_type == 'flooding', flooding_type
                    loss = (loss - flooding_threshold).abs() + flooding_threshold
            
            if loss_max_indices is not None:
                multipliers = torch.ones(loss.shape, dtype=torch.float32, device=loss.device)
                for i, idx in enumerate(ex_idx):
                    if idx in loss_max_indices:
                        multipliers[i] = -1
                loss = loss * multipliers

        loss_values.append(loss.detach().clone())

        assert loss.shape == (len(data),)
        loss = loss.mean()  # Reduction has been disabled -- do explicit reduction
        if gradient_ascent:
            loss = -loss

        if use_autocast:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if log_predictions:
            predictions.append(output.argmax(dim=1).detach())
            example_idx.append(ex_idx.clone())
            targets.append(target.clone())

        if batch_idx % log_interval == 0:
            pbar.set_description(f"Loss: {float(loss):.4f}")
        torch.cuda.synchronize()
    pbar.close()

    output_dict = {}
    if log_predictions:
        # Collect the statistics from all the GPUs
        example_idx = torch.cat(dist_utils.gather_tensor(torch.cat(example_idx, dim=0)), dim=0).detach().cpu().numpy()
        predictions = torch.cat(dist_utils.gather_tensor(torch.cat(predictions, dim=0)), dim=0).detach().cpu().numpy()
        targets = torch.cat(dist_utils.gather_tensor(torch.cat(targets, dim=0)), dim=0).detach().cpu().numpy()
        loss_values = torch.cat(dist_utils.gather_tensor(torch.cat(loss_values, dim=0)), dim=0).detach().cpu().numpy()
        output_dict = {"ex_idx": example_idx, "preds": predictions, "targets": targets, "loss": loss_values}

    return output_dict


# In[ ]:


def test(model, device, criterion, test_loader, distributed, rank, set_name="Test", log_predictions=False):
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

    if isinstance(test_loader.sampler, torch.utils.data.distributed.DistributedSampler):
        num_dataset_ex = len(test_loader.sampler.dataset)
    elif isinstance(test_loader.sampler, DistributedSamplerWrapper):
        num_dataset_ex = len(test_loader.sampler.sampler.dataset)
    elif isinstance(test_loader.sampler, torch.utils.data.SubsetRandomSampler):
        num_dataset_ex = len(test_loader.sampler)
    else:
        # assert test_loader.sampler is None, test_loader.sampler
        num_dataset_ex = len(test_loader.dataset)

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


# In[ ]:


def test_tensor(model, device, criterion, data, target, msg=None, log_predictions=False, batch_size=None):
    assert torch.is_tensor(data) and torch.is_tensor(target)

    model.eval()
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
            pred_list = []
            
            num_batches = int(np.ceil(len(data) / float(batch_size)))
            for i in range(num_batches):
                start, end = i * batch_size, (i+1) * batch_size
                output = model(data[start:end])
                loss_vals = criterion(output, target[start:end])
                loss_vals_list.append(loss_vals.detach())
                
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                pred_list.append(pred.detach())
                correct += pred.eq(target[start:end].view_as(pred)).sum().item()
                total += len(output)
            
            loss_vals = torch.cat(loss_vals_list, dim=0)
            pred = torch.cat(pred_list, dim=0)
            test_loss = float(loss_vals.mean())

    test_acc = 100. * correct / total
    output_dict = dict(loss=test_loss, acc=test_acc, correct=correct, total=total)

    loss_vals = loss_vals.detach().cpu().numpy()
    output_dict["loss_mean"] = np.mean(loss_vals)
    output_dict["loss_var"] = np.var(loss_vals)
    output_dict["loss_std"] = np.std(loss_vals)

    pred_dict = None
    if log_predictions:
        pred_dict = {}
        pred_dict["ex_idx"] = np.arange(len(loss_vals))
        pred_dict["loss_vals"] = loss_vals
        pred_dict["preds"] = pred.detach().cpu().numpy()
        pred_dict["targets"] = target.detach().cpu().numpy()

    header = "Test set" if msg is None else msg
    print(f"{header} | Loss mean: {output_dict['loss_mean']:.4f} | Loss std: {output_dict['loss_std']:.4f} | Accuracy: {test_acc:.2f}% ({correct}/{total})")

    return output_dict, pred_dict


class FreqCNN(torch.nn.Module):
    def __init__(self, image_shape):
        """
            image_shape: [c, h, w]
        """
        super(FreqCNN, self).__init__()

        self.conv1 = torch.nn.Conv2d(image_shape[0], 32, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.elu1 = torch.nn.ELU()

        self.conv2 = torch.nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(32)
        self.elu2 = torch.nn.ELU()

        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=2)
        self.dropout1 = torch.nn.Dropout2d(p=0.2)

        self.conv3 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(64)
        self.elu3 = torch.nn.ELU()

        self.conv4 = torch.nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = torch.nn.BatchNorm2d(64)
        self.elu4 = torch.nn.ELU()

        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=2)
        self.dropout2 = torch.nn.Dropout2d(p=0.3)

        self.conv5 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = torch.nn.BatchNorm2d(128)
        self.elu5 = torch.nn.ELU()

        self.conv6 = torch.nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn6 = torch.nn.BatchNorm2d(128)
        self.elu6 = torch.nn.ELU()

        self.maxpool3 = torch.nn.MaxPool2d(kernel_size=2)
        self.dropout3 = torch.nn.Dropout2d(p=0.4)

        self.flatten = torch.nn.Flatten()

        # TODO: Make this adjust to image size...
        self.fc1 = torch.nn.Linear((image_shape[1] // 2 // 2 // 2) * (image_shape[2] // 2 // 2 // 2) * 128, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.elu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.elu2(x)

        x = self.maxpool1(x)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.elu3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.elu4(x)

        x = self.maxpool2(x)
        x = self.dropout2(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.elu5(x)

        x = self.conv6(x)
        x = self.bn6(x)
        x = self.elu6(x)

        x = self.maxpool3(x)
        x = self.dropout3(x)

        x = self.flatten(x)
        x = self.fc1(x)

        return x
