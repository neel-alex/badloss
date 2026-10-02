"""Shared experiment state and steps: data, probes and poisons, loaders, (re)training and evaluation."""
import json
import os
import subprocess
import warnings

import numpy as np
import torch
from tqdm import tqdm

import dist_utils
import utils
from backdoors import make_probe_imgs, make_poison_imgs, make_poison_imgs_test, all_poison_indices
from dataset_utils import get_settings_for_dataset, make_probe_dataset, make_index_dataset, get_loader
from detection_metrics import get_confusion_stats, get_auc
from plot_utils import plot_probe_examples
from torch_utils import get_model, get_optimizer, train, test, test_tensor


def _git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       cwd=os.path.dirname(os.path.abspath(__file__))).strip()
    except Exception:
        return None


def _jsonable(x):
    if torch.is_tensor(x):
        return x.tolist()
    if isinstance(x, (np.generic, np.ndarray)):
        return x.tolist()
    raise TypeError(f"Not JSON serializable: {type(x)}")


class Experiment:
    """Builds the poisoned training set (with the defender's clean probes) and holds everything the defenses share.

    Setup order matters: it fixes the sequence of draws from the global RNGs, so it is kept as is.
    """

    def __init__(self, args):
        self.args = args
        print(args.dataset, args.attack, args.defense, args.poisoning_ratio)
        self.attacks = args.attacks
        self.seed = args.seed
        print("Seed:", self.seed)
        utils.seed_all(self.seed)

        project_id = "exp68"
        ratio_tag = '_' + str(args.poisoning_ratio)
        self.output_dir = f"./backdoor_{project_id}_{args.dataset}_{args.defense}_{args.attack}{ratio_tag}"
        self.model_collection_dir = (f"./backdoor_{project_id}_model_{args.dataset}_{args.attack}"
                                     f"{'_' + args.defense if args.defense == 'badloss' else ''}{ratio_tag}")

        self.log_wandb = False
        if args.wandb and dist_utils.is_main_proc():
            import wandb
            print("Initializing w&b")
            wandb.init(project=f"badloss_{args.dataset}",
                       name=f"attack_{args.attack}_defense_{args.defense}_poisoning_ratio{args.poisoning_ratio}")
            self.log_wandb = True
        self.wandb_prefix = ''

        self._init_distributed()
        if self.main_proc:
            os.makedirs(self.model_collection_dir, exist_ok=True)
            os.makedirs(self.output_dir, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Current device:", self.device)
        self.results = {"args": vars(args), "git_sha": _git_sha()}

        self._setup_data()

    def _init_distributed(self):
        self.distributed = int(os.getenv('WORLD_SIZE', 1)) > 1
        self.rank = int(os.getenv('RANK', 0))
        self.local_rank = 0
        if "SLURM_NNODES" in os.environ:
            self.local_rank = self.rank % torch.cuda.device_count()
            print(f"SLURM tasks/nodes: {os.getenv('SLURM_NTASKS', 1)}/{os.getenv('SLURM_NNODES', 1)}")
        elif "WORLD_SIZE" in os.environ:
            self.local_rank = int(os.getenv('LOCAL_RANK', 0))

        if self.distributed:
            torch.cuda.set_device(self.local_rank)
            torch.distributed.init_process_group(backend="nccl", init_method="env://")
            world_size = torch.distributed.get_world_size()
            assert int(os.getenv('WORLD_SIZE', 1)) == world_size
            print(f"Initializing the environment with {world_size} processes | Current process rank: {self.local_rank}")

        self.main_proc = dist_utils.is_main_proc(self.local_rank, shared_fs=True)
        print("Is main proc?", self.main_proc)
        dist_utils.setup_for_distributed(self.main_proc)
        warnings.filterwarnings("ignore", "Warning: Leaking Caffe2 thread-pool after fork. (function pthreadpool)",
                                UserWarning)

    def _setup_data(self):
        args, device = self.args, self.device
        (self.img_size, self.train_transform, _, self.no_transform, _, self.train_set, self.train_set_wo_aug,
         self.test_set) = get_settings_for_dataset(args.dataset, args.data_dir)
        print(args.dataset, len(self.train_set), len(self.test_set))

        self.num_classes = len(self.train_set.classes)
        expected_classes = {'cifar10': 10, 'imagenette': 10, 'gtsrb': 43, 'imagenet': 1000}[args.dataset]
        assert self.num_classes == expected_classes, self.num_classes
        print(args.dataset, self.num_classes)

        self.poison_imgs, self.attack_targets, blend_r_pattern = make_poison_imgs(
            self.num_classes, args.dataset, self.train_set_wo_aug, args.poison_ratios, self.attacks,
            self.output_dir, self.main_proc, self.img_size, data_root=args.data_dir)
        self.probe_imgs, unused_probe_imgs = make_probe_imgs(
            self.num_classes, args.dataset, self.train_set_wo_aug, args.num_train_probes, self.output_dir,
            self.main_proc, self.img_size, poison_indices=all_poison_indices(self.poison_imgs))
        self.poison_imgs_test = make_poison_imgs_test(self.test_set, args.dataset, args.num_test_probes, self.attacks,
                                                      self.attack_targets, blend_r_pattern, self.output_dir,
                                                      self.main_proc, self.img_size, data_root=args.data_dir)

        # BaDLoss adds its probes to the training set (as separate examples); other defenses don't
        self.probe_sets = []
        if args.defense == "badloss":
            self.probe_sets = [('unused_probe', unused_probe_imgs), ('probe', self.probe_imgs)]
            plot_probe_examples(self.probe_imgs, {'unused_probe': unused_probe_imgs, **self.poison_imgs},
                                args.dataset, self.train_set, self.rank, self.output_dir, log_wandb=self.log_wandb)
        # Image sets in the training set, evaluated during training
        self.train_eval_sets = {**dict(self.probe_sets[:1]), **self.poison_imgs, **dict(self.probe_sets[1:])}
        # Original training-set positions of the probes added to the training set
        self.probe_original_idx = np.concatenate([image_set.idx for _, image_set in self.probe_sets]
                                                 ) if self.probe_sets else np.array([], dtype=int)

        self.num_epochs = args.num_epochs
        self.batch_size = args.batch_size

        (self.comb_train_set, self.comb_train_indices, self.dataset_probe_identity,
         discarded_idx) = make_probe_dataset(self.train_set, args.dataset, self.train_transform, self.probe_sets,
                                             self.poison_imgs, self.output_dir)
        self.valid_idx = [i for i in range(len(self.train_set)) if i not in discarded_idx]

        # Created for every defense (even those that train their own models) to keep the RNG stream fixed
        self.model = self.new_model()
        self.criterion, self.optimizer, self.lr_scheduler = self.new_optimizer(self.model, self.num_epochs)

        self.new_idx_loader, self.new_idx_loader_wo_aug, self.test_idx_loader, self.idx_dataset = \
            make_index_dataset(self.comb_train_set, self.comb_train_indices, self.test_set, self.no_transform,
                               self.batch_size, self.distributed, args.num_workers, self.seed)

        # Probe/poison sets are small: keep them on the GPU (the test ones too, except for large images)
        for image_set in self.train_eval_sets.values():
            image_set.to(device)
        if args.dataset not in {"gtsrb", "imagenette", "imagenet"}:
            for image_set in self.poison_imgs_test.values():
                image_set.to(device)

        self.model_dir = os.path.join(self.model_collection_dir, f"models_{args.dataset}")
        if self.main_proc:
            os.makedirs(self.model_dir, exist_ok=True)

    # --- Building blocks ---

    def new_model(self):
        return get_model(self.args.dataset, self.num_classes, self.device, self.local_rank, arch=self.args.arch)

    def new_optimizer(self, model, num_epochs):
        return get_optimizer(model, self.device, self.args.lr, self.args.weight_decay, num_epochs)

    def loader(self, indices, dataset=None):
        """Shuffled loader over the given indices of the combined training set (with training augmentation)."""
        return get_loader(self.idx_dataset if dataset is None else dataset, self.distributed, self.args.num_workers,
                          self.seed, indices=indices, batch_size=self.batch_size)

    # --- Evaluation and results ---

    def evaluate(self, model, criterion, train_loader=None):
        """Evaluate on the test set, the training set (un-augmented unless train_loader is given) and each probe
        set; returns test stats."""
        train_loader = self.new_idx_loader_wo_aug if train_loader is None else train_loader
        test_stats, _ = test(model, self.device, criterion, self.test_idx_loader, self.distributed, self.rank)
        train_stats, _ = test(model, self.device, criterion, train_loader, self.distributed, self.rank,
                              set_name="Train")
        self.wandb_log({self.wandb_prefix + "test": test_stats, self.wandb_prefix + "train": train_stats})

        for name, image_set in self.train_eval_sets.items():
            stats = test_tensor(model, self.device, criterion, image_set.images, image_set.labels,
                                msg=f"{name.capitalize().replace('_', ' ')} (train)",
                                batch_size=self.args.eval_batch_size)
            self.wandb_log({self.wandb_prefix + name: stats})
        return test_stats

    def evaluate_asr(self, model, criterion):
        """Attack success rate of each attack on triggered test images (non-target classes)."""
        output_dict = {}
        for attack, image_set in self.poison_imgs_test.items():
            stats = test_tensor(model, self.device, criterion, image_set.images, image_set.labels,
                                msg=f"{attack.capitalize().replace('_', ' ')} ASR (test)",
                                batch_size=self.args.eval_batch_size)
            output_dict[attack] = {'accuracy': stats['acc'], 'total': stats['total'], 'correct': stats['correct']}
        self.wandb_log({"unseen_probes": output_dict})
        return output_dict

    def report_attacked_model(self):
        """Evaluate the model trained on the poisoned data (before any defense is applied)."""
        print("Final model performance:")
        test_stats, _ = test(self.model, self.device, self.criterion, self.test_idx_loader, self.distributed,
                             self.rank)
        asr = self.evaluate_asr(self.model, self.criterion)
        self.record_model_metrics("attacked", test_stats, asr)

    def wandb_log(self, data):
        if self.log_wandb:
            import wandb
            wandb.log(data)

    def save_results(self):
        if self.main_proc:
            with open(os.path.join(self.output_dir, "results.json"), "w") as f:
                json.dump(self.results, f, indent=2, default=_jsonable)

    def record_model_metrics(self, key, test_stats, asr_dict):
        self.results[key] = {"clean_acc": test_stats["acc"],
                             "asr": {k: v["accuracy"] for k, v in asr_dict.items()}}
        self.save_results()

    def record_detection(self, **kwargs):
        self.results.setdefault("detection", {}).update(kwargs)
        self.save_results()

    def report_detection_auc(self, name, idx_list):
        """AUC of a detector from its flagged sets over a sweep of thresholds."""
        det_auc = get_auc(idx_list, self.valid_idx, self.dataset_probe_identity)
        print(f"{name} AUC", det_auc)
        self.record_detection(auc=det_auc)

    def report_detection(self, identified_indices):
        per_class_det = {}
        fpr, tpr = get_confusion_stats(identified_indices, self.valid_idx, self.dataset_probe_identity,
                                       per_class_out=per_class_det)
        self.record_detection(fpr=fpr, tpr=tpr, num_removed=len(identified_indices), detection_rate=per_class_det)

    def finish(self):
        if self.log_wandb:
            import wandb
            wandb.finish()

    # --- Training ---

    def pretrain(self):
        """Train self.model on the poisoned training set (the attacked model inspected by most defenses)."""
        args = self.args
        model_file = os.path.join(self.model_dir, f"model_{args.dataset}_dynamics.pth")
        if not os.path.exists(model_file):
            for epoch in range(self.num_epochs):
                train(self.model, self.device, self.new_idx_loader, self.optimizer, self.criterion)
                if (epoch + 1) % 5 == 0:
                    print(f"Stats for epoch {epoch + 1}")
                    self.evaluate(self.model, self.criterion)
                    self.evaluate_asr(self.model, self.criterion)
                self.lr_scheduler.step()
            if self.main_proc:
                torch.save(self.model.state_dict(), model_file)
        else:
            print("Data files already found. Loading data from saved checkpoints.")
            self.model.load_state_dict(torch.load(model_file, map_location=self.device))

    def retrain(self, identified_indices, checkpoint_tag):
        """Train a fresh model on the combined training set minus the identified examples; report its metrics."""
        print(f"Retraining with {identified_indices.shape[0]} elements removed.")
        retrain_indices = [x for x in self.comb_train_indices if x not in identified_indices]
        retrain_set_dl = self.loader(retrain_indices)

        clean_model = self.new_model()
        clean_criterion, clean_optimizer, clean_lr_scheduler = self.new_optimizer(clean_model, self.num_epochs)

        output_checkpoint_dir = os.path.join(self.output_dir, "model_ft")
        if self.main_proc:
            os.makedirs(output_checkpoint_dir, exist_ok=True)
        output_checkpoint = os.path.join(output_checkpoint_dir, f"model_ft_{checkpoint_tag:.1f}.pth")

        print("Selected output checkpoint:", output_checkpoint)
        if not os.path.exists(output_checkpoint):
            print("!! Output checkpoint not found. Training model from scratch...")
            for epoch in range(self.num_epochs):
                train(clean_model, self.device, retrain_set_dl, clean_optimizer, clean_criterion)
                if (epoch + 1) % 5 == 0:
                    print(f"Stats for epoch {epoch + 1}")
                    self.evaluate(clean_model, clean_criterion)
                clean_lr_scheduler.step()
            torch.save(clean_model.state_dict(), output_checkpoint)
        else:
            print("!! Loading model from pretrained checkpoint:", output_checkpoint)
            clean_model.load_state_dict(torch.load(output_checkpoint, map_location=self.device))

        print("Retrained model performance:")
        test_stats = self.evaluate(clean_model, clean_criterion, train_loader=retrain_set_dl)
        asr = self.evaluate_asr(clean_model, clean_criterion)
        self.record_model_metrics("retrained", test_stats, asr)
        return clean_model

    # --- Features ---

    def last_layer_activations(self, model, loader, masking_op=None):
        """Penultimate-layer activations over a loader.

        masking_op: optional transform applied to each batch first (used by Neural Cleanse).
        Returns tensors: activations (N x D), example indices, true classes, predicted classes.
        """
        activations = {}

        def hook(module, input, output):
            activations['out'] = output.detach()

        # NB: assumes the second-to-last child is the feature layer (true for the ResNets)
        _, layer = list(model.named_children())[-2]
        handle = layer.register_forward_hook(hook)

        all_acts, example_indices, classes, class_preds = [], [], [], []
        for (data, target), ex_idx in tqdm(loader):
            data = data.to(self.device)
            if masking_op is not None:
                data = masking_op(data)
            output = model(data)
            all_acts.append(activations['out'].squeeze())  # Remove size 1 dimensions
            example_indices.append(ex_idx)
            classes.append(target)
            class_preds.append(torch.argmax(output, 1))

        handle.remove()
        return (torch.vstack(all_acts), torch.hstack(example_indices),
                torch.hstack(classes), torch.hstack(class_preds))
