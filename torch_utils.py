import numpy as np
import torch
from torchvision import models
from tqdm import tqdm
try:
    from catalyst.data import DistributedSamplerWrapper
except ImportError:  # Only needed for distributed runs
    DistributedSamplerWrapper = None

import dist_utils


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

    if isinstance(test_loader.sampler, torch.utils.data.distributed.DistributedSampler):
        num_dataset_ex = len(test_loader.sampler.dataset)
    elif DistributedSamplerWrapper is not None and isinstance(test_loader.sampler, DistributedSamplerWrapper):
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

class FeatureExtractor:
    def __init__(self, model):
        self.model = model
        self.last_layer_activations = None
        
        # Register a forward hook on the last layer (layer4)
        self.model.avgpool.register_forward_hook(self.hook)
    
    def hook(self, module, input, output):
        self.last_layer_activations = output
    
    def __call__(self, x):
        logits = self.model(x)
        return logits, torch.flatten(self.last_layer_activations, 1)


def train_cbd(clean_model, backdoor_model, discriminator, device, new_idx_loader, optimizer, adv_optimizer, criterion, ce_gamma=1.0):
    backdoor_model.eval()
    backdoor_model.to(device)
    clean_model.train()
    clean_model.to(device)
    discriminator.train()
    discriminator.to(device)

    backdoor_model_fe, clean_model_fe = FeatureExtractor(backdoor_model), FeatureExtractor(clean_model) # output shape: ([batch x 10], [batch x 2048])

    pbar = tqdm(new_idx_loader)
    for (data, target), ex_idx in pbar:
        data = data.to(device)
        output1, z_hidden = clean_model_fe(data)
        with torch.no_grad():
            output2, r_hidden = backdoor_model_fe(data)
        
        r_hidden, z_hidden = r_hidden.detach(), z_hidden.detach()
        # max dis_loss
        dis_loss = - discriminator(r_hidden, z_hidden)
        adv_optimizer.zero_grad()
        dis_loss.backward()
        adv_optimizer.step()
        # Lipschitz constrain for Disc of WGAN
        discriminator.spectral_norm()
        pbar.set_description(f"Loss: {float(dis_loss.detach()):.4f}")

    pbar = tqdm(new_idx_loader)
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

        weight = (loss_bias / (loss_d + loss_bias + 1e-8)) ** ce_gamma

        weight = weight * weight.shape[0] / torch.sum(weight)
        loss = torch.mean(weight * criterion(output1, target))

        loss += dis_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_description(f"Loss: {float(loss.detach()):.4f}")

def train_intraclass(backdoor_model, device, new_idx_loader, optimizer, num_classes):
    backdoor_model.train()
    backdoor_model.to(device)

    backdoor_model_fe = FeatureExtractor(backdoor_model)

    pbar = tqdm(new_idx_loader)
    for (data, target), ex_idx in pbar:
        data = data.to(device)
        target = target.to(device)

        outputs, features = backdoor_model_fe(data)

        # Calculate intra-class loss
        centers = []
        for j in range(num_classes):
            j_idx = torch.where(target == j)[0]
            if j_idx.shape[0] == 0:
                continue
            j_features = features[j_idx]
            j_center = torch.mean(j_features, dim=0)
            centers.append(j_center)

        centers = torch.stack(centers, dim=0)
        centers = torch.nn.functional.normalize(centers, dim=1)
        similarity_matrix = torch.matmul(centers, centers.T)
        mask = torch.eye(similarity_matrix.shape[0], dtype=torch.bool).to(device)
        similarity_matrix[mask] = 0.0
        loss = torch.mean(similarity_matrix)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def calc_fct(backdoor_model, device, new_idx_loader_wo_aug):
    backdoor_model.eval()
    backdoor_model.to(device)

    backdoor_model_fe = FeatureExtractor(backdoor_model)

    from torchvision import transforms
    fct_transform = transforms.Compose([
        transforms.RandomRotation(180),
        transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),
    ])

    fcts = torch.zeros((len(new_idx_loader_wo_aug.dataset),)).to(device)

    pbar = tqdm(new_idx_loader_wo_aug)
    for (data, target), ex_idx in pbar:
        data = data.to(device)
        target = target.to(device)
        data2 = fct_transform(data)
        with torch.no_grad():
            outputs1, features1 = backdoor_model_fe(data)
            outputs2, features2 = backdoor_model_fe(data2)

        feature_consistency = torch.mean((features1 - features2)**2, dim=1)
        fcts[ex_idx] = feature_consistency
    
    return fcts

def pss_unlearn(model, device, clean_dl, pois_dl, optimizer, criterion):
    model.train()
    model.to(device)

    pbar = tqdm(pois_dl)
    for (data, target), ex_idx in pbar:
        data = data.to(device)
        target = target.to(device)


        output = model(data)
        loss = criterion(output, target)
        
        loss = loss.mean()  # Reduction has been disabled -- do explicit reduction
        
        loss = -loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_description(f"Loss: {float(loss.detach()):.4f}")

    pbar = tqdm(clean_dl)
    for (data, target), ex_idx in pbar:
        data = data.to(device)
        target = target.to(device)


        output = model(data)
        loss = criterion(output, target)
        
        loss = loss.mean()  # Reduction has been disabled -- do explicit reduction

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_description(f"Loss: {float(loss.detach()):.4f}")


def collect_losses(model, device, new_idx_loader_wo_aug, criterion):
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
            # TODO: Report avg. training loss/correct

    return loss_array, probs_array
