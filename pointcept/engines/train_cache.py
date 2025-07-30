"""
GPU Cached Trainer - Extension of your existing trainer with VRAM caching

Author: Extended from Xiaoyang Wu's trainer
"""

import os
import sys
import weakref
import wandb
import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
import time
from packaging import version
from functools import partial
from pathlib import Path
from collections import defaultdict
import psutil
import gc

if sys.version_info >= (3, 10):
    from collections.abc import Iterator
else:
    from collections import Iterator
from tensorboardX import SummaryWriter

from .defaults import create_ddp_model, worker_init_fn
from .hooks import HookBase, build_hooks
import pointcept.utils.comm as comm
from pointcept.datasets import build_dataset, point_collate_fn, collate_fn
from pointcept.models import build_model
from pointcept.utils.logger import get_root_logger
from pointcept.utils.optimizer import build_optimizer
from pointcept.utils.scheduler import build_scheduler
from pointcept.utils.events import EventStorage, ExceptionWriter
from pointcept.utils.registry import Registry

# Import your existing trainer
from .trainer import TRAINERS, TrainerBase, AMP_DTYPE


class GPUCacheManager:
    """
    Manages GPU VRAM caching for training data
    """

    def __init__(self, cache_config, logger=None):
        self.config = cache_config
        self.logger = logger or get_root_logger()
        self.cache = {}
        self.cache_stats = {"hits": 0, "misses": 0, "loads": 0}
        self.memory_stats = {"peak_allocated": 0, "peak_reserved": 0}

        # Cache configuration
        self.enabled = cache_config.get("enabled", True)
        self.cache_mode = cache_config.get(
            "cache_mode", "partial"
        )  # 'full', 'partial', 'adaptive'
        self.cache_ratio = cache_config.get("cache_ratio", 0.7)
        self.gpu_devices = cache_config.get("gpu_devices", [0, 1])
        self.max_cache_memory = cache_config.get("max_cache_memory_gb", 60) * (
            1024**3
        )  # Convert to bytes
        self.preload_transforms = cache_config.get("preload_transforms", True)
        self.prefetch_queue_size = cache_config.get("prefetch_queue_size", 4)

        # Runtime state
        self.current_gpu = 0
        self.cache_indices = set()
        self.access_counts = defaultdict(int)
        self.last_access_time = defaultdict(float)
        self.prefetch_queue = []
        self.cache_memory_usage = {gpu: 0 for gpu in self.gpu_devices}

        if self.enabled:
            self._initialize_cache_memory_limits()

    def _initialize_cache_memory_limits(self):
        """Calculate memory limits per GPU"""
        self.memory_per_gpu = self.max_cache_memory // len(self.gpu_devices)
        self.logger.info(
            f"Cache memory limit per GPU: {self.memory_per_gpu / (1024**3):.2f} GB"
        )

        # Get available memory on each GPU
        for gpu_id in self.gpu_devices:
            try:
                torch.cuda.set_device(gpu_id)
                total_memory = torch.cuda.get_device_properties(gpu_id).total_memory
                allocated = torch.cuda.memory_allocated(gpu_id)
                available = total_memory - allocated
                self.logger.info(
                    f"GPU {gpu_id} - Total: {total_memory/(1024**3):.2f}GB, "
                    f"Available: {available/(1024**3):.2f}GB"
                )
            except Exception as e:
                self.logger.warning(f"Could not get memory info for GPU {gpu_id}: {e}")

    def should_cache_sample(self, idx, sample_size_estimate=None):
        """Determine if a sample should be cached"""
        if not self.enabled:
            return False

        # Check if already cached
        if idx in self.cache_indices:
            return False

        # Check memory constraints
        if sample_size_estimate:
            gpu_id = self.gpu_devices[self.current_gpu % len(self.gpu_devices)]
            if (
                self.cache_memory_usage[gpu_id] + sample_size_estimate
                > self.memory_per_gpu
            ):
                return False

        return True

    def cache_sample(self, idx, sample_data):
        """Cache a sample to GPU memory"""
        if not self.should_cache_sample(idx):
            return False

        try:
            gpu_id = self.gpu_devices[self.current_gpu % len(self.gpu_devices)]

            with torch.cuda.device(gpu_id):
                cached_sample = {}
                sample_memory = 0

                for key, value in sample_data.items():
                    if isinstance(value, torch.Tensor):
                        # Move tensor to GPU
                        cached_tensor = value.cuda(gpu_id, non_blocking=True)
                        cached_sample[key] = cached_tensor
                        sample_memory += (
                            cached_tensor.nelement() * cached_tensor.element_size()
                        )
                    elif isinstance(value, np.ndarray):
                        # Convert numpy to tensor and move to GPU
                        tensor_value = torch.from_numpy(value)
                        cached_tensor = tensor_value.cuda(gpu_id, non_blocking=True)
                        cached_sample[key] = cached_tensor
                        sample_memory += (
                            cached_tensor.nelement() * cached_tensor.element_size()
                        )
                    else:
                        # Keep non-tensor data as-is
                        cached_sample[key] = value

                # Store in cache
                self.cache[idx] = {
                    "data": cached_sample,
                    "gpu_id": gpu_id,
                    "memory_size": sample_memory,
                    "timestamp": time.time(),
                }

                self.cache_indices.add(idx)
                self.cache_memory_usage[gpu_id] += sample_memory
                self.cache_stats["loads"] += 1

                # Round-robin GPU assignment
                self.current_gpu = (self.current_gpu + 1) % len(self.gpu_devices)

                return True

        except Exception as e:
            self.logger.warning(f"Failed to cache sample {idx}: {e}")
            return False

    def get_cached_sample(self, idx):
        """Retrieve a cached sample"""
        if idx not in self.cache_indices:
            self.cache_stats["misses"] += 1
            return None

        try:
            cached_item = self.cache[idx]
            self.cache_stats["hits"] += 1
            self.access_counts[idx] += 1
            self.last_access_time[idx] = time.time()

            # Move to current GPU if needed
            current_device = torch.cuda.current_device()
            if cached_item["gpu_id"] != current_device:
                # Create a copy on current device
                moved_sample = {}
                for key, value in cached_item["data"].items():
                    if isinstance(value, torch.Tensor):
                        moved_sample[key] = value.to(current_device, non_blocking=True)
                    else:
                        moved_sample[key] = value
                return moved_sample
            else:
                return cached_item["data"]

        except Exception as e:
            self.logger.warning(f"Failed to retrieve cached sample {idx}: {e}")
            self.cache_stats["misses"] += 1
            return None

    def evict_lru_samples(self, target_memory_per_gpu):
        """Evict least recently used samples to free memory"""
        for gpu_id in self.gpu_devices:
            if self.cache_memory_usage[gpu_id] <= target_memory_per_gpu:
                continue

            # Get samples on this GPU sorted by last access time
            gpu_samples = [
                (idx, info)
                for idx, info in self.cache.items()
                if info["gpu_id"] == gpu_id
            ]
            gpu_samples.sort(key=lambda x: self.last_access_time[x[0]])

            # Evict oldest samples
            for idx, info in gpu_samples:
                if self.cache_memory_usage[gpu_id] <= target_memory_per_gpu:
                    break

                self.cache_memory_usage[gpu_id] -= info["memory_size"]
                del self.cache[idx]
                self.cache_indices.remove(idx)

                self.logger.debug(f"Evicted sample {idx} from GPU {gpu_id}")

    def get_cache_stats(self):
        """Get cache performance statistics"""
        total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = (
            self.cache_stats["hits"] / total_requests if total_requests > 0 else 0
        )

        return {
            "hit_rate": hit_rate,
            "hits": self.cache_stats["hits"],
            "misses": self.cache_stats["misses"],
            "loads": self.cache_stats["loads"],
            "cached_samples": len(self.cache_indices),
            "memory_usage_gb": {
                gpu: usage / (1024**3) for gpu, usage in self.cache_memory_usage.items()
            },
            "total_memory_gb": sum(self.cache_memory_usage.values()) / (1024**3),
        }

    def clear_cache(self):
        """Clear all cached data"""
        self.cache.clear()
        self.cache_indices.clear()
        self.cache_memory_usage = {gpu: 0 for gpu in self.gpu_devices}
        torch.cuda.empty_cache()
        self.logger.info("Cache cleared")


class CachedDatasetWrapper:
    """
    Wrapper for existing datasets to add caching capability
    """

    def __init__(self, base_dataset, cache_manager, transform=None):
        self.base_dataset = base_dataset
        self.cache_manager = cache_manager
        self.transform = transform
        self.preload_completed = False

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        # Try to get from cache first
        cached_data = self.cache_manager.get_cached_sample(idx)
        if cached_data is not None:
            return cached_data

        # Cache miss - load from base dataset
        data = self.base_dataset[idx]

        # Apply transforms if specified
        if self.transform is not None:
            data = self.transform(data)

        # Try to cache the processed data
        self.cache_manager.cache_sample(idx, data)

        return data

    def preload_cache(self, indices=None, batch_size=32):
        """Preload specified indices into cache"""
        if indices is None:
            dataset_size = len(self.base_dataset)
            if self.cache_manager.cache_mode == "full":
                indices = list(range(dataset_size))
            elif self.cache_manager.cache_mode == "partial":
                cache_count = int(dataset_size * self.cache_manager.cache_ratio)
                indices = list(range(cache_count))
            else:  # adaptive
                indices = self._select_adaptive_indices()

        self.cache_manager.logger.info(
            f"Preloading {len(indices)} samples into cache..."
        )

        # Preload in batches to avoid memory spikes
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i : i + batch_size]
            for idx in batch_indices:
                try:
                    _ = self[idx]  # This will trigger caching
                except Exception as e:
                    self.cache_manager.logger.warning(
                        f"Failed to preload sample {idx}: {e}"
                    )

            # Log progress
            if (i + batch_size) % (batch_size * 10) == 0:
                stats = self.cache_manager.get_cache_stats()
                self.cache_manager.logger.info(
                    f"Preloaded {min(i + batch_size, len(indices))}/{len(indices)} samples. "
                    f"Cache memory: {stats['total_memory_gb']:.2f}GB"
                )

        self.preload_completed = True
        final_stats = self.cache_manager.get_cache_stats()
        self.cache_manager.logger.info(
            f"Preloading completed. Cached {final_stats['cached_samples']} samples, "
            f"using {final_stats['total_memory_gb']:.2f}GB"
        )

    def _select_adaptive_indices(self):
        """Select indices for adaptive caching (prioritize smaller files, validation data, etc.)"""
        dataset_size = len(self.base_dataset)
        cache_count = int(dataset_size * self.cache_manager.cache_ratio)

        # For now, just return first N samples
        # You can implement more sophisticated selection logic here
        return list(range(cache_count))


@TRAINERS.register_module("CachedTrainer")
class CachedTrainer(TrainerBase):
    """
    Enhanced trainer with GPU VRAM caching support
    """

    def __init__(self, cfg):
        super(CachedTrainer, self).__init__()
        self.epoch = 0
        self.start_epoch = 0
        self.max_epoch = cfg.eval_epoch
        self.best_metric_value = -torch.inf
        self.logger = get_root_logger(
            log_file=os.path.join(cfg.save_path, "train.log"),
            file_mode="a" if cfg.resume else "w",
        )
        self.logger.info("=> Loading config ...")
        self.cfg = cfg
        self.logger.info(f"Save path: {cfg.save_path}")
        self.logger.info(f"Config:\n{cfg.pretty_text}")

        # Initialize cache manager
        cache_config = getattr(cfg.data, "gpu_cache", {})
        self.cache_manager = GPUCacheManager(cache_config, self.logger)

        self.logger.info("=> Building model ...")
        self.model = self.build_model()
        self.logger.info("=> Building writer ...")
        self.writer = self.build_writer()
        self.logger.info("=> Building cached train dataset & dataloader ...")
        self.train_loader = self.build_train_loader()
        self.logger.info("=> Building cached val dataset & dataloader ...")
        self.val_loader = self.build_val_loader()
        self.logger.info("=> Building optimize, scheduler, scaler(amp) ...")
        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_scheduler()
        self.scaler = self.build_scaler()
        self.logger.info("=> Building hooks ...")
        self.register_hooks(self.cfg.hooks)

        # Preload cache if enabled
        if self.cache_manager.enabled and hasattr(self, "cached_train_dataset"):
            self.logger.info("=> Preloading training cache ...")
            self.cached_train_dataset.preload_cache()
        if self.cache_manager.enabled and hasattr(self, "cached_val_dataset"):
            self.logger.info("=> Preloading validation cache ...")
            self.cached_val_dataset.preload_cache()

    def train(self):
        with EventStorage() as self.storage, ExceptionWriter():
            # => before train
            self.before_train()
            self.logger.info(">>>>>>>>>>>>>>>> Start Training >>>>>>>>>>>>>>>>")

            # Log initial cache stats
            if self.cache_manager.enabled:
                initial_stats = self.cache_manager.get_cache_stats()
                self.logger.info(f"Initial cache stats: {initial_stats}")

            for self.epoch in range(self.start_epoch, self.max_epoch):
                # => before epoch
                if comm.get_world_size() > 1:
                    self.train_loader.sampler.set_epoch(self.epoch)
                self.model.train()
                self.data_iterator = enumerate(self.train_loader)
                self.before_epoch()

                # => run_epoch
                epoch_start_time = time.time()
                for (
                    self.comm_info["iter"],
                    self.comm_info["input_dict"],
                ) in self.data_iterator:
                    # => before_step
                    self.before_step()
                    # => run_step
                    self.run_step()
                    # => after_step
                    self.after_step()

                    # Log cache stats periodically
                    if (
                        self.cache_manager.enabled
                        and self.comm_info["iter"] % 100 == 0
                        and comm.is_main_process()
                    ):
                        stats = self.cache_manager.get_cache_stats()
                        self.logger.info(
                            f"Iter {self.comm_info['iter']}: "
                            f"Cache hit rate: {stats['hit_rate']:.3f}"
                        )

                # => after epoch
                epoch_time = time.time() - epoch_start_time
                if self.cache_manager.enabled and comm.is_main_process():
                    stats = self.cache_manager.get_cache_stats()
                    self.logger.info(
                        f"Epoch {self.epoch} completed in {epoch_time:.2f}s. "
                        f"Cache stats: {stats}"
                    )

                self.after_epoch()
            # => after train
            self.after_train()

    def run_step(self):
        if version.parse(torch.__version__) >= version.parse("2.4"):
            auto_cast = partial(torch.amp.autocast, device_type="cuda")
        else:
            auto_cast = torch.cuda.amp.autocast

        input_dict = self.comm_info["input_dict"]

        # Move data to GPU (data might already be on GPU from cache)
        for key in input_dict.keys():
            if isinstance(input_dict[key], torch.Tensor):
                if not input_dict[key].is_cuda:
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)

        with auto_cast(
            enabled=self.cfg.enable_amp, dtype=AMP_DTYPE[self.cfg.amp_dtype]
        ):
            output_dict = self.model(input_dict)
            loss = output_dict["loss"]

        self.optimizer.zero_grad()
        if self.cfg.enable_amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            if self.cfg.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.clip_grad
                )
            self.scaler.step(self.optimizer)

            scaler = self.scaler.get_scale()
            self.scaler.update()
            if scaler <= self.scaler.get_scale():
                self.scheduler.step()
        else:
            loss.backward()
            if self.cfg.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.clip_grad
                )
            self.optimizer.step()
            self.scheduler.step()

        # Memory management
        if self.cfg.empty_cache:
            torch.cuda.empty_cache()

        # Periodic cache cleanup
        if self.cache_manager.enabled and self.comm_info["iter"] % 500 == 0:
            self._manage_cache_memory()

        self.comm_info["model_output_dict"] = output_dict

    def _manage_cache_memory(self):
        """Manage cache memory usage"""
        try:
            # Check current memory usage
            for gpu_id in self.cache_manager.gpu_devices:
                allocated = torch.cuda.memory_allocated(gpu_id)
                reserved = torch.cuda.memory_reserved(gpu_id)
                total = torch.cuda.get_device_properties(gpu_id).total_memory

                # If memory usage is too high, evict some cache
                if reserved > total * 0.9:  # 90% threshold
                    target_cache_memory = self.cache_manager.memory_per_gpu * 0.8
                    self.cache_manager.evict_lru_samples(target_cache_memory)
                    self.logger.info(
                        f"Evicted cache on GPU {gpu_id} due to high memory usage"
                    )
        except Exception as e:
            self.logger.warning(f"Error managing cache memory: {e}")

    def build_model(self):
        model = build_model(self.cfg.model)
        if self.cfg.sync_bn:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"Num params: {n_parameters}")
        model = create_ddp_model(
            model.cuda(),
            broadcast_buffers=False,
            find_unused_parameters=self.cfg.find_unused_parameters,
        )
        return model

    def build_writer(self):
        writer = SummaryWriter(self.cfg.save_path) if comm.is_main_process() else None
        self.logger.info(f"Tensorboard writer logging dir: {self.cfg.save_path}")
        if self.cfg.enable_wandb and comm.is_main_process():
            tag, name = Path(self.cfg.save_path).parts[-2:]
            wandb.init(
                project=self.cfg.wandb_project,
                name=f"{tag}/{name}",
                tags=[tag],
                dir=self.cfg.save_path,
                settings=wandb.Settings(api_key=self.cfg.wandb_key),
                config=self.cfg,
            )
        return writer

    def build_train_loader(self):
        # Build base dataset
        train_data = build_dataset(self.cfg.data.train)

        # Wrap with caching if enabled
        if self.cache_manager.enabled:
            self.cached_train_dataset = CachedDatasetWrapper(
                train_data, self.cache_manager
            )
            train_data = self.cached_train_dataset

        if comm.get_world_size() > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        else:
            train_sampler = None

        init_fn = (
            partial(
                worker_init_fn,
                num_workers=self.cfg.num_worker_per_gpu,
                rank=comm.get_rank(),
                seed=self.cfg.seed,
            )
            if self.cfg.seed is not None
            else None
        )

        # Adjust num_workers for cached data (can use fewer workers)
        num_workers = self.cfg.num_worker_per_gpu
        if self.cache_manager.enabled:
            num_workers = max(2, num_workers // 2)  # Use fewer workers for cached data

        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=self.cfg.batch_size_per_gpu,
            shuffle=(train_sampler is None),
            num_workers=num_workers,
            sampler=train_sampler,
            collate_fn=partial(point_collate_fn, mix_prob=self.cfg.mix_prob),
            pin_memory=not self.cache_manager.enabled,  # No need to pin if already on GPU
            worker_init_fn=init_fn,
            drop_last=len(train_data) > self.cfg.batch_size,
            persistent_workers=True,
            prefetch_factor=4 if self.cache_manager.enabled else 2,
        )
        return train_loader

    def build_val_loader(self):
        val_loader = None
        if self.cfg.evaluate:
            val_data = build_dataset(self.cfg.data.val)

            # Wrap with caching if enabled (usually cache all validation data)
            if self.cache_manager.enabled:
                val_cache_config = self.cache_manager.config.copy()
                val_cache_config["cache_mode"] = "full"  # Cache all validation data
                val_cache_manager = GPUCacheManager(val_cache_config, self.logger)
                self.cached_val_dataset = CachedDatasetWrapper(
                    val_data, val_cache_manager
                )
                val_data = self.cached_val_dataset

            if comm.get_world_size() > 1:
                val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
            else:
                val_sampler = None

            val_loader = torch.utils.data.DataLoader(
                val_data,
                batch_size=self.cfg.batch_size_val_per_gpu,
                shuffle=False,
                num_workers=(
                    2 if self.cache_manager.enabled else self.cfg.num_worker_per_gpu
                ),
                pin_memory=not self.cache_manager.enabled,
                sampler=val_sampler,
                collate_fn=collate_fn,
            )
        return val_loader

    def build_optimizer(self):
        return build_optimizer(self.cfg.optimizer, self.model, self.cfg.param_dicts)

    def build_scheduler(self):
        assert hasattr(self, "optimizer")
        assert hasattr(self, "train_loader")
        self.cfg.scheduler.total_steps = len(self.train_loader) * self.cfg.eval_epoch
        return build_scheduler(self.cfg.scheduler, self.optimizer)

    def build_scaler(self):
        if version.parse(torch.__version__) >= version.parse("2.4"):
            grad_scaler = partial(torch.amp.GradScaler, device="cuda")
        else:
            grad_scaler = torch.cuda.amp.GradScaler
        scaler = grad_scaler() if self.cfg.enable_amp else None
        return scaler

    def after_epoch(self):
        for h in self.hooks:
            h.after_epoch()
        self.storage.reset_histories()

        # Cache memory management at end of epoch
        if self.cache_manager.enabled:
            self._manage_cache_memory()

        if self.cfg.empty_cache_per_epoch:
            torch.cuda.empty_cache()

    def after_train(self):
        # Log final cache statistics
        if self.cache_manager.enabled and comm.is_main_process():
            final_stats = self.cache_manager.get_cache_stats()
            self.logger.info(f"Final cache statistics: {final_stats}")

        super().after_train()

        # Clean up cache
        if self.cache_manager.enabled:
            self.cache_manager.clear_cache()
