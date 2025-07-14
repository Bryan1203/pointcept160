"""
Main Training Script with GPU VRAM Caching Support

Author: Modified from Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Enhanced for H100 dual-GPU training with VRAM caching
"""

import os
import sys
import time
import torch
import psutil
import logging
from contextlib import contextmanager

from pointcept.engines.defaults import (
    default_argument_parser,
    default_config_parser,
    default_setup,
)
from pointcept.engines.train_cache import TRAINERS
from pointcept.engines.launch import launch
import pointcept.utils.comm as comm


def setup_gpu_memory_monitoring():
    """Setup GPU memory monitoring for cache optimization"""
    if not torch.cuda.is_available():
        return
    
    # Enable memory history for debugging
    torch.cuda.memory._record_memory_history(True, trace_alloc_max_entries=100000)
    
    # Set memory fraction if specified in environment
    memory_fraction = os.environ.get('CUDA_MEMORY_FRACTION', None)
    if memory_fraction:
        torch.cuda.set_per_process_memory_fraction(float(memory_fraction))


def log_system_info(logger):
    """Log system and GPU information"""
    logger.info("=" * 60)
    logger.info("SYSTEM INFORMATION")
    logger.info("=" * 60)
    
    # CPU and RAM info
    cpu_count = psutil.cpu_count()
    memory = psutil.virtual_memory()
    logger.info(f"CPU cores: {cpu_count}")
    logger.info(f"System RAM: {memory.total / (1024**3):.1f} GB")
    logger.info(f"Available RAM: {memory.available / (1024**3):.1f} GB")
    
    # GPU info
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            logger.info(f"GPU {i}: {props.name}")
            logger.info(f"  Total memory: {props.total_memory / (1024**3):.1f} GB")
            logger.info(f"  Compute capability: {props.major}.{props.minor}")
            
            # Current memory usage
            if torch.cuda.is_initialized():
                allocated = torch.cuda.memory_allocated(i) / (1024**3)
                reserved = torch.cuda.memory_reserved(i) / (1024**3)
                logger.info(f"  Current allocated: {allocated:.2f} GB")
                logger.info(f"  Current reserved: {reserved:.2f} GB")
    
    logger.info("=" * 60)


def validate_cache_config(cfg):
    """Validate GPU cache configuration"""
    if not hasattr(cfg.data, 'gpu_cache') or not cfg.data.gpu_cache.get('enabled', False):
        return True
    
    cache_config = cfg.data.gpu_cache
    gpu_devices = cache_config.get('gpu_devices', [0])
    max_cache_memory_gb = cache_config.get('max_cache_memory_gb', 60)
    
    # Check if specified GPUs exist
    available_gpus = torch.cuda.device_count()
    for gpu_id in gpu_devices:
        if gpu_id >= available_gpus:
            raise ValueError(f"GPU {gpu_id} specified in cache config but only {available_gpus} GPUs available")
    
    # Check memory requirements
    for gpu_id in gpu_devices:
        props = torch.cuda.get_device_properties(gpu_id)
        total_memory_gb = props.total_memory / (1024**3)
        
        if max_cache_memory_gb > total_memory_gb * 0.9:
            print(f"WARNING: Cache memory ({max_cache_memory_gb}GB) > 90% of GPU {gpu_id} "
                  f"memory ({total_memory_gb:.1f}GB). This may cause OOM errors.")
    
    return True


@contextmanager
def gpu_memory_profiler(logger, operation_name="Operation"):
    """Context manager for GPU memory profiling"""
    if not torch.cuda.is_available():
        yield
        return
    
    # Record initial state
    initial_allocated = {}
    initial_reserved = {}
    for i in range(torch.cuda.device_count()):
        initial_allocated[i] = torch.cuda.memory_allocated(i)
        initial_reserved[i] = torch.cuda.memory_reserved(i)
    
    start_time = time.time()
    
    try:
        yield
    finally:
        end_time = time.time()
        
        # Log memory changes
        logger.info(f"{operation_name} completed in {end_time - start_time:.2f}s")
        for i in range(torch.cuda.device_count()):
            final_allocated = torch.cuda.memory_allocated(i)
            final_reserved = torch.cuda.memory_reserved(i)
            
            allocated_diff = (final_allocated - initial_allocated[i]) / (1024**3)
            reserved_diff = (final_reserved - initial_reserved[i]) / (1024**3)
            
            if abs(allocated_diff) > 0.1 or abs(reserved_diff) > 0.1:  # Only log significant changes
                logger.info(f"GPU {i} memory change - "
                           f"Allocated: {allocated_diff:+.2f}GB, "
                           f"Reserved: {reserved_diff:+.2f}GB")


def precompute_cache_estimates(cfg, logger):
    """Estimate cache requirements before training"""
    if not hasattr(cfg.data, 'gpu_cache') or not cfg.data.gpu_cache.get('enabled', False):
        return
    
    logger.info("Estimating cache requirements...")
    
    try:
        # Build a sample dataset to estimate memory requirements
        from pointcept.datasets import build_dataset
        
        # Build training dataset
        train_dataset = build_dataset(cfg.data.train)
        dataset_size = len(train_dataset)
        
        # Sample a few items to estimate memory usage
        sample_indices = [0, min(10, dataset_size-1), min(100, dataset_size-1)]
        total_sample_size = 0
        
        for idx in sample_indices:
            try:
                sample = train_dataset[idx]
                sample_size = 0
                
                for key, value in sample.items():
                    if isinstance(value, torch.Tensor):
                        sample_size += value.nelement() * value.element_size()
                    elif hasattr(value, 'nbytes'):  # numpy array
                        sample_size += value.nbytes
                
                total_sample_size += sample_size
                
            except Exception as e:
                logger.warning(f"Could not sample item {idx} for estimation: {e}")
                continue
        
        if len(sample_indices) > 0:
            avg_sample_size = total_sample_size / len(sample_indices)
            cache_ratio = cfg.data.gpu_cache.get('cache_ratio', 0.6)
            estimated_cache_size = (dataset_size * cache_ratio * avg_sample_size) / (1024**3)
            
            logger.info(f"Dataset size: {dataset_size} samples")
            logger.info(f"Average sample size: {avg_sample_size / (1024**2):.2f} MB")
            logger.info(f"Cache ratio: {cache_ratio:.1%}")
            logger.info(f"Estimated cache size: {estimated_cache_size:.2f} GB")
            
            # Check if cache will fit
            num_gpus = len(cfg.data.gpu_cache.get('gpu_devices', [0]))
            cache_per_gpu = estimated_cache_size / num_gpus
            max_cache_per_gpu = cfg.data.gpu_cache.get('max_cache_memory_gb', 60)
            
            if cache_per_gpu > max_cache_per_gpu:
                logger.warning(f"Estimated cache ({cache_per_gpu:.2f}GB per GPU) exceeds "
                              f"configured limit ({max_cache_per_gpu}GB per GPU)")
                # Suggest adjusted cache ratio
                suggested_ratio = (max_cache_per_gpu * num_gpus) / estimated_cache_size * cache_ratio
                logger.warning(f"Consider reducing cache_ratio to {suggested_ratio:.2f}")
            else:
                logger.info(f"Cache should fit: {cache_per_gpu:.2f}GB per GPU "
                           f"(limit: {max_cache_per_gpu}GB)")
        
    except Exception as e:
        logger.warning(f"Could not estimate cache requirements: {e}")


def main_worker(cfg):
    cfg = default_setup(cfg)
    logger = logging.getLogger(__name__)
    
    # Setup GPU memory monitoring
    setup_gpu_memory_monitoring()
    
    # Log system information
    if comm.is_main_process():
        log_system_info(logger)
    
    # Validate cache configuration
    if comm.is_main_process():
        validate_cache_config(cfg)
        precompute_cache_estimates(cfg, logger)
    
    # Synchronize before building trainer
    if comm.get_world_size() > 1:
        comm.synchronize()
    
    # Build trainer with memory profiling
    with gpu_memory_profiler(logger, "Trainer initialization"):
        trainer = TRAINERS.build(dict(type=cfg.train.type, cfg=cfg))
    
    # Log final memory state before training
    if comm.is_main_process() and torch.cuda.is_available():
        logger.info("Pre-training GPU memory state:")
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / (1024**3)
            reserved = torch.cuda.memory_reserved(i) / (1024**3)
            total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            utilization = (reserved / total) * 100
            logger.info(f"GPU {i}: {allocated:.2f}GB allocated, "
                       f"{reserved:.2f}GB reserved, "
                       f"{utilization:.1f}% utilization")
    
    # Start training with memory profiling
    try:
        with gpu_memory_profiler(logger, "Training"):
            trainer.train()
    except torch.cuda.OutOfMemoryError as e:
        logger.error("CUDA Out of Memory Error!")
        logger.error("Try reducing one of the following:")
        logger.error("- batch_size")
        logger.error("- gpu_cache.cache_ratio")
        logger.error("- gpu_cache.max_cache_memory_gb")
        logger.error("- model size")
        raise e
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise e
    finally:
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def enhanced_argument_parser():
    """Enhanced argument parser with cache-specific options"""
    parser = default_argument_parser()
    
    # Add cache-specific arguments
    parser.add_argument(
        "--cache-mode",
        type=str,
        choices=["disabled", "partial", "full", "adaptive"],
        help="Override cache mode in config",
    )
    
    parser.add_argument(
        "--cache-ratio",
        type=float,
        help="Override cache ratio in config (0.0-1.0)",
    )
    
    parser.add_argument(
        "--max-cache-memory",
        type=float,
        help="Override max cache memory per GPU in GB",
    )
    
    parser.add_argument(
        "--profile-memory",
        action="store_true",
        help="Enable detailed memory profiling",
    )
    
    parser.add_argument(
        "--estimate-cache-only",
        action="store_true",
        help="Only estimate cache requirements and exit",
    )
    
    return parser


def apply_argument_overrides(cfg, args):
    """Apply command line argument overrides to config"""
    if not hasattr(cfg.data, 'gpu_cache'):
        cfg.data.gpu_cache = {}
    
    if args.cache_mode:
        if args.cache_mode == "disabled":
            cfg.data.gpu_cache.enabled = False
        else:
            cfg.data.gpu_cache.enabled = True
            cfg.data.gpu_cache.cache_mode = args.cache_mode
    
    if args.cache_ratio is not None:
        cfg.data.gpu_cache.cache_ratio = args.cache_ratio
    
    if args.max_cache_memory is not None:
        cfg.data.gpu_cache.max_cache_memory_gb = args.max_cache_memory
    
    return cfg


def main():
    parser = enhanced_argument_parser()
    args = parser.parse_args()
    cfg = default_config_parser(args.config_file, args.options)
    
    # Apply command line overrides
    cfg = apply_argument_overrides(cfg, args)
    
    # Handle estimation-only mode
    if args.estimate_cache_only:
        print("Estimating cache requirements...")
        logger = logging.getLogger(__name__)
        validate_cache_config(cfg)
        precompute_cache_estimates(cfg, logger)
        return
    
    # Set memory profiling
    if args.profile_memory:
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    
    # Launch training
    launch(
        main_worker,
        num_gpus_per_machine=args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        cfg=(cfg,),
    )


if __name__ == "__main__":
    main()