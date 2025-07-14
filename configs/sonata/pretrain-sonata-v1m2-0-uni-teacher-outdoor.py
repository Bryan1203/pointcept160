"""
Default configuration for pretraining a SONATA model
Dataset: NuScenes, SemenaticKITTI, Waymo
"""

_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
batch_size = 6  # bs: total bs in all gpus (3 per H100, same ratio as indoor)
num_worker = 12  # Scaled down from 96 for 4 GPUs
mix_prob = 0
clip_grad = 3.0
empty_cache = False
enable_amp = True
amp_dtype = "bfloat16"
evaluate = False
find_unused_parameters = False


# model settings
model = dict(
    type="Sonata-v1m2",
    # backbone - student & teacher
    backbone=dict(
        type="PT-v3m2",
        in_channels=4,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(3, 3, 3, 12, 3),
        enc_channels=(48, 96, 192, 384, 512),
        enc_num_head=(3, 6, 12, 24, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        traceable=True,
        enc_mode=True,
        mask_token=True,
    ),
    teacher_custom=dict(
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
    ),
    head_in_channels=1088,
    head_hidden_channels=4096,
    head_embed_channels=256,
    head_num_prototypes=4096,
    num_global_view=2,
    num_local_view=4,
    mask_size_start=0.1,
    mask_size_base=0.4,
    mask_size_warmup_ratio=0.05,
    mask_ratio_start=0.3,
    mask_ratio_base=0.7,
    mask_ratio_warmup_ratio=0.05,
    mask_jitter=0.01,
    teacher_temp_start=0.04,
    teacher_temp_base=0.07,
    teacher_temp_warmup_ratio=0.05,
    student_temp=0.1,
    mask_loss_weight=2 / 8,
    roll_mask_loss_weight=2 / 8,
    unmask_loss_weight=4 / 8,
    momentum_base=0.994,
    momentum_final=1,
    match_max_k=8,
    match_max_r=0.32,
    up_cast_level=2,
)

# scheduler settings
# epoch = 200
epoch = 20
eval_epoch = 20 
base_lr = 0.0014  # Scaled with sqrt(12/96) = sqrt(1/8) from original 0.004
lr_decay = 0.9  # layer-wise lr decay

base_wd = 0.04  # wd scheduler enable in hooks
final_wd = 0.2  # wd scheduler enable in hooks

dec_depths = model["backbone"]["enc_depths"]
param_dicts = [
    dict(
        keyword=f"enc{e}.block{b}.",
        lr=base_lr * lr_decay ** (sum(dec_depths) - sum(dec_depths[:e]) - b - 1),
    )
    for e in range(len(dec_depths))
    for b in range(dec_depths[e])
]
del dec_depths

optimizer = dict(type="AdamW", lr=base_lr, weight_decay=base_wd)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[base_lr] + [g["lr"] for g in param_dicts],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)

# dataset settings
transform = [
    dict(type="GridSample", grid_size=0.05, hash_type="fnv", mode="train"),
    dict(type="Copy", keys_dict={"coord": "origin_coord"}),
    dict(
        type="MultiViewGenerator",
        view_keys=("coord", "origin_coord", "strength"),  # Removed color and normal
        global_view_num=2,
        global_view_scale=(0.6, 1.0),
        local_view_num=4,
        local_view_scale=(0.2, 0.6),
        global_shared_transform=[
            # Add intensity augmentation similar to your ITRI config
            dict(type='IntensityAugmentation', 
                scale_range=(0.9, 1.1),  # More conservative for pretraining
                shift_range=(-2.0, 2.0),  # Smaller range than supervised
                noise_std=0.5,
                gamma_range=(0.9, 1.1),
                augment_prob=0.5),  # Lower prob for pretraining
            dict(type="RandomDropout", dropout_ratio=0.1, dropout_application_ratio=0.3),  # Reduced intensity
        ],
        global_transform=[
            dict(type="CenterShift", apply_z=False),  # Don't shift Z for outdoor
            # dict(type="RandomScale", scale=[0.95, 1.05]),  # Removed - inappropriate for road scenes
            dict(type="RandomRotate", angle=[-0.5, 0.5], axis="z", center=[0, 0, 0], p=0.7),  # More rotation
            dict(type="RandomRotate", angle=[-1 / 128, 1 / 128], axis="x", p=0.3),
            dict(type="RandomRotate", angle=[-1 / 128, 1 / 128], axis="y", p=0.3),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.003, clip=0.015),  # Slightly reduced jitter
        ],
        local_transform=[
            dict(type="CenterShift", apply_z=False),
            # dict(type="RandomScale", scale=[0.9, 1.1]),  # Removed - inappropriate for road scenes
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.7),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.3),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.3),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            # Additional intensity augmentation for local views
            dict(type='IntensityAugmentation', 
                scale_range=(0.8, 1.2),  # More aggressive for local views
                shift_range=(-1.0, 1.0),
                noise_std=0.3,
                gamma_range=(0.9, 1.1),
                augment_prob=0.3),
        ],
        max_size=65536,  # Increased from 32768 - good balance for 139K point clouds
    ),
    dict(type="ToTensor"),
    dict(type="Update", keys_dict={"grid_size": 0.05}),  # Keep consistent with initial grid_size
    dict(
        type="Collect",
        keys=(
            "global_origin_coord",
            "global_coord",
            "global_strength",  # Changed from global_color
            "global_offset",
            "local_origin_coord",
            "local_coord", 
            "local_strength",   # Changed from local_color
            "local_offset",
            "grid_size",
            "name",
        ),
        offset_keys_dict=dict(),
        global_feat_keys=("global_coord", "global_strength"),  # Only coord + strength
        local_feat_keys=("local_coord", "local_strength"),     # Only coord + strength
    ),
]
data_roots = [
    '/data2/itri464058/pointcloud_data/hsinchu_q2/',
    '/data2/itri464058/pointcloud_data/airport_q2/'
]

domain_names = ['hsinchu_q2', 'airport_q2']

data = dict(
    num_classes=12,
    ignore_index=-1,
    names=['none', 'solid', 'broken', 'solid_solid', 'solid_broken', 'broken_solid', 'broken_broken', 'botts_dots', 'grass', 'curb', 'custom', 'edge'],
    
    train=dict(
        type='ItriDataset',
        split='test',
        data_root=data_roots,  # Single domain first
        domain_names=domain_names,
        test_mode=False,
        ignore_index=-1,
        loop=1,
        transform=transform,  # Use the Sonata transforms defined above
    )
)

# trainer settings (using DefaultTrainer for single dataset)
train = dict(type="DefaultTrainer")

hooks = [
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="WeightDecaySchedular", base_value=base_wd, final_value=final_wd),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="CheckpointSaver", save_freq=5),
]
