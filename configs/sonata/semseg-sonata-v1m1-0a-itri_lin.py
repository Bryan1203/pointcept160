"""
Linear probing configuration for SONATA model on ITRI dataset
- Uses pretrained SONATA-v1m2 backbone from outdoor pretraining
- Multi-domain training on hsinchu_q2 and airport_q2 datasets
- 12 classes for road marking segmentation
- Input: 4 channels (coord(3) + strength(1))
- Frozen backbone with trainable segmentation head
"""

_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
batch_size = 24  # bs: total bs in all gpus
num_worker = 48
mix_prob = 0.8
clip_grad = 3.0
empty_cache = False
enable_amp = True
amp_dtype = "bfloat16"

# model settings
model = dict(
    type="DefaultSegmentorV2",
    num_classes=12,
    backbone_out_channels=64,  # Should be 64 with proper decoder
    backbone=dict(
        type="PT-v3m2",
        in_channels=4,  # Match ITRI-pretrained model (coord(3) + strength(1))
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(3, 3, 3, 12, 3),
        enc_channels=(48, 96, 192, 384, 512),
        enc_num_head=(3, 6, 12, 24, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),  # Added missing decoder
        dec_channels=(64, 96, 192, 384),  # Added missing decoder
        dec_num_head=(4, 6, 12, 24),  # Added missing decoder
        dec_patch_size=(1024, 1024, 1024, 1024),  # Added missing decoder
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
        traceable=False,  # Changed for fine-tuning
        mask_token=False,  # Changed for fine-tuning
        enc_mode=False,  # Changed for fine-tuning
        freeze_encoder=False,
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
    freeze_backbone=False,
)

# scheduler settings
epoch = 80
eval_epoch = 80
optimizer = dict(type="AdamW", lr=0.002, weight_decay=0.02)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.002, 0.0002],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
param_dicts = [dict(keyword="block", lr=0.0002)]

# Multi-domain data configuration
data_roots = [
    "/data2/itri464058/pointcloud_data/hsinchu_q2/",
    "/data2/itri464058/pointcloud_data/airport_q2/",
]

domain_names = ["hsinchu_q2", "airport_q2"]

data = dict(
    num_classes=12,
    ignore_index=-1,
    names=[
        "none",
        "solid",
        "broken",
        "solid_solid",
        "solid_broken",
        "broken_solid",
        "broken_broken",
        "botts_dots",
        "grass",
        "curb",
        "custom",
        "edge",
    ],
    # Multi-domain training
    train=dict(
        type="ItriDataset",
        split="train",
        data_root=data_roots,  # Pass list for multi-domain
        domain_names=domain_names,
        domain_balance=False,  # Balance samples across domains
        transform=[
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            # Intensity augmentation
            dict(
                type="IntensityAugmentation",
                scale_range=(0.8, 1.2),
                shift_range=(-3.0, 3.0),
                noise_std=1.0,
                gamma_range=(0.8, 1.2),
                augment_prob=0.7,
            ),
            dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment"),
                feat_keys=("coord", "strength"),
            ),
        ],
        test_mode=False,
        ignore_index=-1,
        loop=1,
    ),
    # Multi-domain validation
    val=dict(
        type="ItriDataset",
        split="val",
        data_root=data_roots,
        domain_names=domain_names,
        domain_balance=False,  # No balancing for validation
        transform=[
            dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment"),
                feat_keys=("coord", "strength"),
            ),
        ],
        test_mode=False,
        ignore_index=-1,
    ),
    # Multi-domain testing
    test=dict(
        type="ItriDataset",
        split="test",
        data_root=data_roots,
        domain_names=domain_names,
        domain_balance=False,
        transform=[
            dict(type="Copy", keys_dict=dict(segment="origin_segment")),
            dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="train",
                return_inverse=True,
            ),
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="test",
                return_grid_coord=True,
            ),
            crop=None,
            post_transform=[
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "index"),
                    feat_keys=("coord", "strength"),
                ),
            ],
            aug_transform=[
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[0],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ]
            ],
        ),
        ignore_index=-1,
    ),
)


# hooks
hooks = [
    dict(
        type="CheckpointLoader",
        keywords="module.student.backbone",
        replacement="module.backbone",
    ),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=False),
]
