import torch
import numpy as np
import os
import pickle
from collections import defaultdict
from torch.utils.data import Dataset
import logging

class GPUCachedItriDataset(Dataset):
    """
    GPU-cached version of ItriDataset for H100 training acceleration
    """
    
    def __init__(self, original_dataset, cache_mode='partial', cache_ratio=0.8, 
                 gpu_devices=None, preload_transforms=True):
        """
        Args:
            original_dataset: Your existing ItriDataset instance
            cache_mode: 'full', 'partial', or 'adaptive'
            cache_ratio: Fraction of data to cache (for partial mode)
            gpu_devices: List of GPU device IDs to distribute cache across
            preload_transforms: Whether to cache transformed data
        """
        self.original_dataset = original_dataset
        self.cache_mode = cache_mode
        self.cache_ratio = cache_ratio
        self.preload_transforms = preload_transforms
        
        # Multi-GPU setup for H100s
        if gpu_devices is None:
            gpu_devices = [0, 1]  # Your dual H100 setup
        self.gpu_devices = gpu_devices
        self.current_device = 0
        
        # Cache storage
        self.gpu_cache = {}
        self.cache_indices = set()
        self.cache_stats = {'hits': 0, 'misses': 0}
        
        # Memory monitoring
        self.max_cache_size_per_gpu = self._calculate_max_cache_size()
        
        # Initialize cache
        self._initialize_cache()
        
    def _calculate_max_cache_size(self):
        """Calculate maximum cache size per GPU (reserve 20% for model/gradients)"""
        total_memory = torch.cuda.get_device_properties(0).total_memory
        # Reserve 20% for model, gradients, and other operations
        available_for_cache = int(total_memory * 0.8)
        # Distribute across GPUs
        return available_for_cache // len(self.gpu_devices)
    
    def _estimate_sample_size(self, sample_idx=0):
        """Estimate memory size of a single sample"""
        try:
            sample = self.original_dataset[sample_idx]
            size = 0
            
            for key, value in sample.items():
                if isinstance(value, torch.Tensor):
                    size += value.nelement() * value.element_size()
                elif isinstance(value, np.ndarray):
                    size += value.nbytes
                    
            return size
        except:
            # Fallback estimate: ~10MB per sample
            return 10 * 1024 * 1024
    
    def _initialize_cache(self):
        """Initialize GPU cache based on cache mode"""
        dataset_size = len(self.original_dataset)
        sample_size = self._estimate_sample_size()
        
        max_samples_per_gpu = self.max_cache_size_per_gpu // sample_size
        total_cacheable = max_samples_per_gpu * len(self.gpu_devices)
        
        if self.cache_mode == 'full':
            cache_count = min(dataset_size, total_cacheable)
        elif self.cache_mode == 'partial':
            cache_count = min(int(dataset_size * self.cache_ratio), total_cacheable)
        else:  # adaptive
            cache_count = min(dataset_size // 2, total_cacheable)
        
        # Select indices to cache (prioritize frequently accessed ones)
        if self.cache_mode == 'adaptive':
            # Cache validation set + random training samples
            cache_indices = self._select_adaptive_indices(cache_count)
        else:
            # Cache first N samples (you might want to randomize this)
            cache_indices = list(range(cache_count))
        
        logging.info(f"Caching {len(cache_indices)} samples across {len(self.gpu_devices)} GPUs")
        logging.info(f"Estimated cache size per GPU: {max_samples_per_gpu * sample_size / (1024**3):.2f} GB")
        
        # Preload cache
        self._preload_cache(cache_indices)
    
    def _select_adaptive_indices(self, cache_count):
        """Select indices for adaptive caching"""
        dataset_size = len(self.original_dataset)
        
        # Always cache validation samples if this is training data
        val_indices = []
        if hasattr(self.original_dataset, 'split') and self.original_dataset.split == 'train':
            # Assume validation is smaller, cache all validation-like indices
            val_count = min(cache_count // 4, dataset_size // 10)
            val_indices = list(range(dataset_size - val_count, dataset_size))
        
        # Fill remaining with random training samples
        remaining_count = cache_count - len(val_indices)
        train_indices = np.random.choice(
            [i for i in range(dataset_size) if i not in val_indices],
            size=min(remaining_count, dataset_size - len(val_indices)),
            replace=False
        ).tolist()
        
        return val_indices + train_indices
    
    def _preload_cache(self, cache_indices):
        """Preload selected samples into GPU cache"""
        samples_per_gpu = len(cache_indices) // len(self.gpu_devices)
        
        for gpu_idx, device_id in enumerate(self.gpu_devices):
            start_idx = gpu_idx * samples_per_gpu
            end_idx = start_idx + samples_per_gpu
            if gpu_idx == len(self.gpu_devices) - 1:  # Last GPU gets remaining samples
                end_idx = len(cache_indices)
            
            gpu_indices = cache_indices[start_idx:end_idx]
            
            with torch.cuda.device(device_id):
                for idx in gpu_indices:
                    try:
                        # Load and transform sample
                        sample = self.original_dataset[idx]
                        
                        # Convert to GPU tensors
                        gpu_sample = {}
                        for key, value in sample.items():
                            if isinstance(value, (torch.Tensor, np.ndarray)):
                                if isinstance(value, np.ndarray):
                                    value = torch.from_numpy(value)
                                gpu_sample[key] = value.cuda(device_id)
                            else:
                                gpu_sample[key] = value
                        
                        # Store in cache with device info
                        self.gpu_cache[idx] = {
                            'data': gpu_sample,
                            'device': device_id
                        }
                        self.cache_indices.add(idx)
                        
                    except Exception as e:
                        logging.warning(f"Failed to cache sample {idx}: {e}")
                        continue
        
        logging.info(f"Successfully cached {len(self.cache_indices)} samples")
    
    def __len__(self):
        return len(self.original_dataset)
    
    def __getitem__(self, idx):
        """Get item with GPU cache lookup"""
        if idx in self.cache_indices:
            # Cache hit
            self.cache_stats['hits'] += 1
            cached_item = self.gpu_cache[idx]
            
            # Move to current device if needed
            current_device = torch.cuda.current_device()
            if cached_item['device'] != current_device:
                # Copy to current device
                data = {}
                for key, value in cached_item['data'].items():
                    if isinstance(value, torch.Tensor):
                        data[key] = value.to(current_device)
                    else:
                        data[key] = value
                return data
            else:
                return cached_item['data']
        else:
            # Cache miss - load from original dataset
            self.cache_stats['misses'] += 1
            return self.original_dataset[idx]
    
    def get_cache_stats(self):
        """Get cache performance statistics"""
        total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
        hit_rate = self.cache_stats['hits'] / total_requests if total_requests > 0 else 0
        return {
            'hit_rate': hit_rate,
            'hits': self.cache_stats['hits'],
            'misses': self.cache_stats['misses'],
            'cached_samples': len(self.cache_indices),
            'total_samples': len(self.original_dataset)
        }
    
    def clear_cache(self):
        """Clear GPU cache to free memory"""
        for cached_item in self.gpu_cache.values():
            del cached_item['data']
        self.gpu_cache.clear()
        self.cache_indices.clear()
        torch.cuda.empty_cache()


class SmartDataLoader:
    """
    Smart DataLoader that optimizes for GPU caching
    """
    
    def __init__(self, dataset, batch_size, num_workers=4, pin_memory=True, 
                 prefetch_factor=2, persistent_workers=True):
        """
        Args:
            dataset: GPUCachedItriDataset instance
            batch_size: Batch size
            num_workers: Number of CPU workers for non-cached data
            pin_memory: Pin memory for faster GPU transfer
            prefetch_factor: Number of batches to prefetch
            persistent_workers: Keep workers alive between epochs
        """
        self.dataset = dataset
        self.batch_size = batch_size
        
        # Configure DataLoader for optimal performance
        self.dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            collate_fn=self._smart_collate
        )
    
    def _smart_collate(self, batch):
        """Smart collation that handles mixed CPU/GPU data"""
        if not batch:
            return {}
        
        # Group by keys
        keys = batch[0].keys()
        collated = {}
        
        for key in keys:
            values = [item[key] for item in batch]
            
            if all(isinstance(v, torch.Tensor) for v in values):
                # Stack tensors
                collated[key] = torch.stack(values)
            elif all(isinstance(v, (str, int, float)) for v in values):
                # Keep as list for non-tensor data
                collated[key] = values
            else:
                # Mixed types - convert to tensor if possible
                try:
                    collated[key] = torch.stack([
                        torch.from_numpy(v) if isinstance(v, np.ndarray) else v 
                        for v in values
                    ])
                except:
                    collated[key] = values
        
        return collated
    
    def __iter__(self):
        return iter(self.dataloader)
    
    def __len__(self):
        return len(self.dataloader)


# Integration with your existing config
def create_cached_datasets(config):
    """
    Create GPU-cached datasets from your config
    """
    # Import your existing dataset class
    from your_dataset_module import ItriDataset  # Adjust import path
    
    # Create original datasets
    train_dataset = ItriDataset(**config['data']['train'])
    val_dataset = ItriDataset(**config['data']['val'])
    
    # Wrap with GPU caching
    cached_train = GPUCachedItriDataset(
        train_dataset, 
        cache_mode='partial',
        cache_ratio=0.6,  # Cache 60% of training data
        gpu_devices=[0, 1]  # Your H100s
    )
    
    cached_val = GPUCachedItriDataset(
        val_dataset,
        cache_mode='full',  # Cache all validation data
        gpu_devices=[0, 1]
    )
    
    # Create optimized data loaders
    train_loader = SmartDataLoader(
        cached_train,
        batch_size=config.get('batch_size', 12),
        num_workers=8,  # Reduce since we're using GPU cache
        persistent_workers=True
    )
    
    val_loader = SmartDataLoader(
        cached_val,
        batch_size=config.get('batch_size_val', config.get('batch_size', 12)),
        num_workers=4,
        persistent_workers=True
    )
    
    return train_loader, val_loader, cached_train, cached_val


# Usage example with monitoring
def train_with_cache_monitoring(model, train_loader, val_loader, cached_train, cached_val):
    """
    Training loop with cache performance monitoring
    """
    for epoch in range(num_epochs):
        # Training
        model.train()
        for batch_idx, batch in enumerate(train_loader):
            # Your training code here
            # ...
            
            # Log cache stats periodically
            if batch_idx % 100 == 0:
                stats = cached_train.get_cache_stats()
                logging.info(f"Epoch {epoch}, Batch {batch_idx}: "
                           f"Cache hit rate: {stats['hit_rate']:.3f}")
        
        # Validation
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                # Your validation code here
                # ...
                pass
        
        # End of epoch cache stats
        train_stats = cached_train.get_cache_stats()
        val_stats = cached_val.get_cache_stats()
        
        logging.info(f"Epoch {epoch} Cache Performance:")
        logging.info(f"  Training: {train_stats['hit_rate']:.3f} hit rate")
        logging.info(f"  Validation: {val_stats['hit_rate']:.3f} hit rate")


# Memory-efficient alternative: Streaming cache
class StreamingGPUCache:
    """
    Streaming cache that maintains a rolling window of recent samples
    """
    
    def __init__(self, max_samples=1000, gpu_devices=[0, 1]):
        self.max_samples = max_samples
        self.gpu_devices = gpu_devices
        self.cache = {}
        self.access_order = []
        self.current_gpu = 0
        
    def get(self, idx, load_fn):
        """Get item from cache or load it"""
        if idx in self.cache:
            # Move to front of access order
            self.access_order.remove(idx)
            self.access_order.append(idx)
            return self.cache[idx]
        
        # Load item
        item = load_fn(idx)
        
        # Add to cache
        self._add_to_cache(idx, item)
        return item
    
    def _add_to_cache(self, idx, item):
        """Add item to cache with LRU eviction"""
        # Convert to GPU tensor
        with torch.cuda.device(self.gpu_devices[self.current_gpu]):
            gpu_item = {}
            for key, value in item.items():
                if isinstance(value, (torch.Tensor, np.ndarray)):
                    if isinstance(value, np.ndarray):
                        value = torch.from_numpy(value)
                    gpu_item[key] = value.cuda()
                else:
                    gpu_item[key] = value
        
        # Evict if necessary
        while len(self.cache) >= self.max_samples:
            oldest_idx = self.access_order.pop(0)
            del self.cache[oldest_idx]
        
        # Add new item
        self.cache[idx] = gpu_item
        self.access_order.append(idx)
        
        # Round-robin GPU assignment
        self.current_gpu = (self.current_gpu + 1) % len(self.gpu_devices)