"""Shared experiment state and steps: data, probes and poisons, loaders, (re)training and evaluation."""
import glob
import hashlib
import json
import os
import random
import subprocess
import time
import warnings

import numpy as np
import torch
from tqdm import tqdm

import dist_utils
import utils
from backdoors import make_probe_imgs, make_poison_imgs, make_poison_imgs_test, all_poison_indices
from dataset_utils import (get_settings_for_dataset, make_probe_dataset, make_index_dataset, get_loader,
                           subsample_per_class)
from defenses.feature_extractor import FeatureExtractor
from detection_metrics import get_confusion_stats, get_auc
from plot_utils import plot_probe_examples
from torch_utils import get_model, get_optimizer, train, test, test_tensor


def _git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       cwd=os.path.dirname(os.path.abspath(__file__))).strip()
    except Exception:
        return None


def _code_hash():
    """Hash of all source files, so cached artifacts are never reused across code changes."""
    root = os.path.dirname(os.path.abspath(__file__))
    files = sorted(glob.glob(os.path.join(root, "*.py")) + glob.glob(os.path.join(root, "defenses", "*.py")))
    h = hashlib.sha256()
    for f in files:
        h.update(os.path.relpath(f, root).encode())
        with open(f, "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()


# Arguments that cannot affect the poisoned training set or the attacked model
_DEFENSE_ARG_PREFIXES = ('badloss_', 'nc_', 'ac_', 'ss_', 'freq_', 'abl_', 'cd_', 'cbd_', 'pss_')
_NON_SEMANTIC_ARGS = {'defense', 'wandb', 'output_root', 'cache_dir', 'no_cache', 'detect_only', 'time_limit_hours',
                      'eval_every'}


class TimeLimitReached(Exception):
    """Raised between epochs when the next one might not finish within --time_limit_hours. Progress is checkpointed:
    rerunning the same command resumes."""


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
        self.start_time = time.time()
        self._max_epoch_time = 0.0  # Longest training epoch so far in this process (for --time_limit_hours)
        self._epochs_run = 0
        print(args.dataset, args.attack, args.defense, args.poisoning_ratio)
        self.attacks = args.attacks
        self.seed = args.seed
        print("Seed:", self.seed)
        utils.seed_all(self.seed)

        ratio_tag = ''.join(f"_{r}" for r in args.poisoning_ratio)
        budget_tag = "_bm" if args.budget_matched_to else ""
        attack_tag = args.attack.replace(',', '+')
        self.output_dir = os.path.join(args.output_root, f"{args.dataset}_{attack_tag}_{args.defense}_s{args.seed}"
                                                         f"{ratio_tag}{budget_tag}")
        self.code_hash = _code_hash()

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
            os.makedirs(self.output_dir, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Current device:", self.device)
        self.results = {"args": dict(vars(args)), "git_sha": _git_sha(), "code_hash": self.code_hash}

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
        if args.train_subset_frac < 1:  # Smoke tests; own RNG, so later draws are unaffected
            rng = np.random.default_rng(self.seed)
            keep = subsample_per_class(self.train_set, args.train_subset_frac, rng)
            assert (subsample_per_class(self.train_set_wo_aug, args.train_subset_frac, np.random.default_rng(self.seed))
                    == keep).all()
            subsample_per_class(self.test_set, args.train_subset_frac, rng)
        print(args.dataset, len(self.train_set), len(self.test_set))

        self.num_classes = len(self.train_set.classes)
        expected_classes = {'cifar10': 10, 'imagenette': 10, 'gtsrb': 43, 'imagenet': 1000}[args.dataset]
        assert self.num_classes == expected_classes, self.num_classes
        print(args.dataset, self.num_classes)

        self.poison_imgs, self.attack_targets, blend_r_pattern = make_poison_imgs(
            self.num_classes, args.dataset, self.train_set_wo_aug, args.poison_ratios, self.attacks,
            self.output_dir, self.main_proc, self.img_size, data_root=args.data_dir,
            budget_matched_to=args.budget_matched_to)
        self.probe_imgs = make_probe_imgs(self.train_set_wo_aug, args.num_probes,
                                          poison_indices=all_poison_indices(self.poison_imgs))
        self.poison_imgs_test = make_poison_imgs_test(self.test_set, args.dataset, args.num_asr_images, self.attacks,
                                                      self.attack_targets, blend_r_pattern, self.output_dir,
                                                      self.main_proc, self.img_size, data_root=args.data_dir)

        # BaDLoss adds its probes to the training set (as separate examples); other defenses don't
        self.probe_sets = []
        if args.defense == "badloss":
            self.probe_sets = [('probe', self.probe_imgs)]
            plot_probe_examples(self.probe_imgs, self.poison_imgs, args.dataset, self.train_set, self.rank,
                                self.output_dir, log_wandb=self.log_wandb)
        # Image sets in the training set, evaluated during training
        self.train_eval_sets = {**self.poison_imgs, **dict(self.probe_sets)}
        # Original training-set positions of the probes added to the training set
        self.probe_original_idx = np.concatenate([image_set.idx for _, image_set in self.probe_sets]
                                                 ) if self.probe_sets else np.array([], dtype=int)

        self.num_epochs = args.num_epochs
        self.batch_size = args.batch_size

        (self.comb_train_set, self.comb_train_indices, self.dataset_probe_identity,
         discarded_idx) = make_probe_dataset(self.train_set, args.dataset, self.train_transform, self.probe_sets,
                                             self.poison_imgs, self.output_dir)
        self.valid_idx = [i for i in range(len(self.train_set)) if i not in discarded_idx]

        # The attacked model (trained on the poisoned set) is created by init_attacked_model(), for defenses using it
        self.model = None
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none').to(device)

        (self.new_idx_loader, self.train_loader_wo_aug, self.new_idx_loader_wo_aug, self.test_idx_loader,
         self.idx_dataset) = \
            make_index_dataset(self.comb_train_set, self.comb_train_indices, self.test_set, self.no_transform,
                               self.batch_size, self.distributed, args.num_workers, self.seed,
                               fix_inserted_aug=args.fix_inserted_aug)

        # Probe/poison sets are small: keep them on the GPU, the ASR test sets too (~8 GB on ImageNet, which would
        # otherwise sit in host memory for the whole run)
        for image_set in list(self.train_eval_sets.values()) + list(self.poison_imgs_test.values()):
            image_set.to(device)


    # --- Building blocks ---

    def target_classes(self):
        return sorted({int(t) for t in self.attack_targets.values()})

    def is_poison(self):
        """Boolean array over the combined training set: which examples are poisons (ground truth, for oracles)."""
        return np.array([x.startswith('poison_') for x in self.dataset_probe_identity])

    def init_attacked_model(self):
        self.model = self.new_model()
        self.criterion, self.optimizer, self.lr_scheduler = self.new_optimizer(self.model, self.num_epochs)

    def new_model(self):
        return get_model(self.args.dataset, self.num_classes, self.device, self.local_rank, arch=self.args.arch,
                         channels_last=self.args.amp)

    def new_optimizer(self, model, num_epochs):
        return get_optimizer(model, self.device, self.args.lr, self.args.weight_decay, num_epochs)

    def cache_dir(self, kind, **extra):
        """Directory for a cached artifact (or None if caching is disabled), keyed by the source code, the
        arguments that can affect it and `extra`. Artifacts are reused only if all of these match."""
        if self.args.no_cache:
            return None
        cache_root = self.args.cache_dir or os.path.join(self.args.output_root, "cache")
        return os.path.join(cache_root, f"{kind}_{self._key(kind, **extra)}")

    def _key(self, kind, **extra):
        """Hash of the source code, the arguments that can affect the attacked model and `extra`."""
        key_args = {k: v for k, v in vars(self.args).items()
                    if k not in _NON_SEMANTIC_ARGS and not k.startswith(_DEFENSE_ARG_PREFIXES)}
        key = json.dumps({"kind": kind, "code": self.code_hash, "args": key_args, "extra": extra},
                         sort_keys=True, default=str)
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    def save_cached(self, cache_dir, **items):
        """Atomically write torch-serializable items into cache_dir (with the key's description)."""
        if cache_dir is None or not self.main_proc:
            return
        os.makedirs(cache_dir, exist_ok=True)
        for name, value in items.items():
            tmp = os.path.join(cache_dir, f".{name}.tmp{os.getpid()}")
            torch.save(value, tmp)
            os.replace(tmp, os.path.join(cache_dir, f"{name}.pt"))
        with open(os.path.join(cache_dir, "args.json"), "w") as f:
            json.dump(self.results["args"], f, indent=2, default=str)

    def load_cached(self, cache_dir, *names):
        """The cached items, or None if any is missing (or caching is disabled)."""
        if cache_dir is None or not all(os.path.exists(os.path.join(cache_dir, f"{n}.pt")) for n in names):
            return None
        print(f"Loading cached {', '.join(names)} from {cache_dir}")
        return [torch.load(os.path.join(cache_dir, f"{n}.pt"), map_location="cpu", weights_only=False)
                for n in names]

    def detect(self, detect_fn):
        """Run a defense's detection step, or load its result from the cache.

        detect_fn() records the detection metrics and returns what retraining needs (e.g. the flagged indices).
        The cache also holds those metrics and the RNG states right after detection, which are restored on a hit,
        so a hit gives exactly the same results as running detect_fn(). Returns None with --detect_only.
        """
        defense_args = {k: v for k, v in vars(self.args).items() if k.startswith(f"{self.args.defense}_")}
        cache_dir = self.cache_dir(f"detection_{self.args.defense}", defense_args=defense_args)
        cached = self.load_cached(cache_dir, "output", "detection", "rng_state")
        if cached is not None:
            output, detection, rng_state = cached
            self.record_detection(**detection)
            self._set_rng_state(rng_state)
        else:
            output = detect_fn()
            self.save_cached(cache_dir, output=output, detection=self.results.get("detection", {}),
                             rng_state=self._get_rng_state())
        return None if self.args.detect_only else output

    def _loaders(self):
        return [self.new_idx_loader, self.train_loader_wo_aug, self.new_idx_loader_wo_aug, self.test_idx_loader]

    def _get_rng_state(self, extra_loaders=()):
        return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "loaders": [loader.generator.get_state() for loader in self._loaders() + list(extra_loaders)]}

    def _set_rng_state(self, state, extra_loaders=()):
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        if state["cuda"] is not None:
            torch.cuda.set_rng_state_all(state["cuda"])
        loaders = self._loaders() + list(extra_loaders)
        assert len(loaders) == len(state["loaders"])
        for loader, loader_state in zip(loaders, state["loaders"]):
            loader.generator.set_state(loader_state)

    def epochs(self, name, num_epochs, state, extra_loaders=()):
        """Epoch numbers 0..num_epochs-1 for a training loop that can be resumed by a later run of the same command.

        After each epoch, checkpoints `state` (name -> a tensor, or an object with state_dict()/load_state_dict()),
        the global RNG states and the loaders' generators (the shared ones plus extra_loaders) to
        <output_dir>/checkpoints/<name>.pt. If that file exists, the loop resumes from it; results are the same as
        an uninterrupted run. With --time_limit_hours, raises TimeLimitReached between epochs when the next epoch
        might not finish in time. `name` must be unique within a run.
        """
        path, key, ckpt = self._load_checkpoint(name, state)
        start, max_epoch_time = 0, self._max_epoch_time
        if ckpt is not None:
            self._set_rng_state(ckpt["rng_state"], extra_loaders)
            start, max_epoch_time = ckpt["epoch"], max(max_epoch_time, ckpt["max_epoch_time"])
            print(f"Resuming {name} from epoch {start} ({path})")
        if start < num_epochs:
            self._check_time_limit(name, start, num_epochs, max_epoch_time)
        for epoch in range(start, num_epochs):
            t0 = time.time()
            yield epoch
            max_epoch_time = self._max_epoch_time = max(max_epoch_time, time.time() - t0)
            self._epochs_run += 1
            self._save_checkpoint(path, key, state, extra_loaders, epoch=epoch + 1, max_epoch_time=max_epoch_time)
            if epoch + 1 < num_epochs:
                self._check_time_limit(name, epoch + 1, num_epochs, max_epoch_time)

    def resumable_pass(self, name, loader, state, num_chunks=100):
        """(batch_idx, batch) for one pass over a sequential loader whose per-batch work is slow (e.g. CD's mask
        optimization), resumable like epochs(): checkpoints `state` and the RNGs every 1/num_chunks of the pass and
        may raise TimeLimitReached between chunks. A resumed run re-reads (but skips) the finished batches, so the
        loader's workers are seeded as in an uninterrupted run; requires the loader's generator to be in the same
        state when this is called (as after reset_rngs)."""
        path, key, ckpt = self._load_checkpoint(name, state)
        done, max_chunk_time = (0, 0.0) if ckpt is None else (ckpt["batch"], ckpt["max_chunk_time"])
        if ckpt is not None:
            print(f"Resuming {name} from batch {done}/{len(loader)} ({path})")
        ends = {round(len(loader) * k / num_chunks) for k in range(1, num_chunks)}
        t0 = time.time()
        for batch_idx, batch in enumerate(loader):
            if batch_idx < done:
                continue
            if batch_idx == done and ckpt is not None:
                self._set_rng_state(ckpt["rng_state"])
                t0 = time.time()
            yield batch_idx, batch
            if batch_idx + 1 in ends:
                max_chunk_time = max(max_chunk_time, time.time() - t0)
                self._epochs_run += 1
                self._save_checkpoint(path, key, state, (), batch=batch_idx + 1, max_chunk_time=max_chunk_time)
                self._check_time_limit(name, batch_idx + 1, len(loader), max_chunk_time)
                t0 = time.time()

    def _load_checkpoint(self, name, state):
        """(path, key, checkpoint or None) for a resumable loop; restores `state` from the checkpoint if there is
        one (the caller restores the RNGs)."""
        path = os.path.join(self.output_dir, "checkpoints", f"{name}.pt")
        defense_args = {k: v for k, v in vars(self.args).items() if k.startswith(_DEFENSE_ARG_PREFIXES)}
        key = self._key(f"checkpoint_{name}", defense=self.args.defense, defense_args=defense_args)
        if not os.path.exists(path):
            return path, key, None
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if ckpt["key"] != key:
            raise RuntimeError(f"{path} is from a different code version or settings; delete it to start over")
        for k, obj in state.items():
            if torch.is_tensor(obj):
                obj.copy_(ckpt["state"][k])
            else:
                obj.load_state_dict(ckpt["state"][k])
        return path, key, ckpt

    def _save_checkpoint(self, path, key, state, extra_loaders, **progress):
        if not self.main_proc:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        ckpt = {"key": key, **progress,
                "state": {k: obj if torch.is_tensor(obj) else obj.state_dict() for k, obj in state.items()},
                "rng_state": self._get_rng_state(extra_loaders)}
        tmp = f"{path}.tmp{os.getpid()}"
        torch.save(ckpt, tmp)
        os.replace(tmp, path)

    def _check_time_limit(self, name, epoch, num_epochs, max_epoch_time):
        """Raise TimeLimitReached if another epoch (taking 1.5x the longest so far) would overrun --time_limit_hours.
        Never before this process has trained an epoch, so every run makes progress."""
        limit, elapsed = self.args.time_limit_hours, time.time() - self.start_time
        if limit is not None and self._epochs_run > 0 and elapsed + 1.5 * max_epoch_time > limit * 3600:
            raise TimeLimitReached(f"{name}: stopping at {epoch}/{num_epochs} ({elapsed / 3600:.2f} h elapsed, the next "
                                   f"step takes up to {max_epoch_time / 60:.1f} min)")

    def reset_rngs(self):
        """Reseed the global RNGs and the loaders after a cacheable phase, so whatever follows is the same
        whether that phase was computed or loaded from the cache."""
        utils.seed_all(self.seed + 1)
        # The sequential loaders' generators seed their workers, which augment the inserted images; they advance
        # with every evaluation pass made while computing the phase (none on a cache hit).
        for loader in (self.new_idx_loader_wo_aug, self.test_idx_loader):
            loader.generator.manual_seed(self.seed)
        self.new_idx_loader = self.loader(self.comb_train_indices)
        self.train_loader_wo_aug = self.loader(self.comb_train_indices, dataset=self.new_idx_loader_wo_aug.dataset)

    def loader(self, indices, dataset=None):
        """Shuffled loader over the given indices of the combined training set (with training augmentation)."""
        return get_loader(self.idx_dataset if dataset is None else dataset, self.distributed, self.args.num_workers,
                          self.seed, indices=indices, batch_size=self.batch_size)

    # --- Evaluation and results ---

    def eval_due(self, epoch):
        """Whether to evaluate mid-training after this (0-based) epoch: every --eval_every epochs."""
        return (epoch + 1) % self.args.eval_every == 0

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

    def save_model(self, model, name):
        if self.main_proc:
            torch.save(model.state_dict(), os.path.join(self.output_dir, name))

    def finish(self):
        """End of a completed run: drop the training checkpoints."""
        if self.main_proc:
            for f in glob.glob(os.path.join(self.output_dir, "checkpoints", "*.pt")):
                os.remove(f)
        if self.log_wandb:
            import wandb
            wandb.finish()

    # --- Training ---

    def pretrain(self):
        """Train the attacked model on the poisoned training set (the model inspected by most defenses).
        Cached across defenses."""
        self.init_attacked_model()
        if self.args.attacked_model is not None:
            self.load_attacked_model(self.args.attacked_model)
            self.reset_rngs()
            return
        cache_dir = self.cache_dir("attacked_model")
        cached = self.load_cached(cache_dir, "model")
        if cached is not None:
            self.model.load_state_dict(cached[0])
        else:
            state = {"model": self.model, "optimizer": self.optimizer, "lr_scheduler": self.lr_scheduler}
            for epoch in self.epochs("pretrain", self.num_epochs, state):
                train(self.model, self.device, self.new_idx_loader, self.optimizer, self.criterion,
                      amp=self.args.amp)
                if self.eval_due(epoch):
                    print(f"Stats for epoch {epoch + 1}")
                    self.evaluate(self.model, self.criterion)
                    self.evaluate_asr(self.model, self.criterion)
                self.lr_scheduler.step()
            self.save_cached(cache_dir, model=self.model.state_dict())
        self.reset_rngs()

    def load_attacked_model(self, path):
        """Load an attacked model's state dict (optionally wrapped as {"model": ...} or saved from DDP)."""
        with open(path, "rb") as f:
            sha256 = hashlib.sha256(f.read()).hexdigest()
        print(f"Loading the attacked model from {path} (sha256 {sha256})")
        state = torch.load(path, map_location="cpu", weights_only=False)
        state = state.get("model", state)
        self.model.load_state_dict({k.removeprefix("module."): v for k, v in state.items()})
        self.results["attacked_model"] = {"path": os.path.abspath(path), "sha256": sha256}

    def retrain(self, identified_indices):
        """Train a fresh model on the combined training set minus the identified examples; report its metrics."""
        print(f"Retraining with {identified_indices.shape[0]} elements removed.")
        comb_train_indices = np.asarray(self.comb_train_indices)
        retrain_indices = comb_train_indices[~np.isin(comb_train_indices, np.asarray(identified_indices))].tolist()
        retrain_set_dl = self.loader(retrain_indices)

        clean_model = self.new_model()
        clean_criterion, clean_optimizer, clean_lr_scheduler = self.new_optimizer(clean_model, self.num_epochs)

        state = {"model": clean_model, "optimizer": clean_optimizer, "lr_scheduler": clean_lr_scheduler}
        for epoch in self.epochs("retrain", self.num_epochs, state, extra_loaders=[retrain_set_dl]):
            train(clean_model, self.device, retrain_set_dl, clean_optimizer, clean_criterion, amp=self.args.amp)
            if self.eval_due(epoch):
                print(f"Stats for epoch {epoch + 1}")
                self.evaluate(clean_model, clean_criterion)
            clean_lr_scheduler.step()
        self.save_model(clean_model, "retrained_model.pth")

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
        fe = FeatureExtractor(model)
        all_acts, example_indices, classes, class_preds = [], [], [], []
        for (data, target), ex_idx in tqdm(loader):
            data = data.to(self.device)
            if masking_op is not None:
                data = masking_op(data)
            output, features = fe(data)
            all_acts.append(features.detach())
            example_indices.append(ex_idx)
            classes.append(target)
            class_preds.append(torch.argmax(output, 1))

        fe.remove()
        return (torch.vstack(all_acts), torch.hstack(example_indices),
                torch.hstack(classes), torch.hstack(class_preds))
