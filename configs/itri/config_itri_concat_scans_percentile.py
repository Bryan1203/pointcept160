_base_ = ["../_base_/default_runtime.py"]

weight = None
resume = False
evaluate = True
test_only = False
seed = 28024989
save_path = "exp/itri/concat_3scans_2025"
num_worker = 12
batch_size = 12
batch_size_val = None
batch_size_test = None
epoch = 50
eval_epoch = 50
sync_bn = False
enable_amp = True
amp_dtype = "float16"
clip_grad = 1.0

empty_cache = False
find_unused_parameters = False
mix_prob = 0.8
param_dicts = [dict(keyword="block", lr=0.002)]

hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=False),
]
train = dict(type="DefaultTrainer")
test = dict(type="SemSegTester", verbose=True)

# Model configuration (standard 4 channels)
model = dict(
    type="DefaultSegmentorV2",
    num_classes=12,
    backbone_out_channels=64,
    backbone=dict(
        type="PT-v3m1",
        in_channels=4,  # coord(3) + strength(1) = 4
        order=["z", "z-trans", "hilbert", "hilbert-trans"],
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(64, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
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
        cls_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("nuScenes", "SemanticKITTI", "Waymo"),
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
)

optimizer = dict(type="AdamW", lr=0.002, weight_decay=0.005)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.002, 0.0002],
    pct_start=0.04,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=100.0,
)

# Multi-domain data configuration with 3-scan concatenation
data_roots = [
    # '/home/bryan/pointcloud_data/hsinchu_q2/',
    # '/home/bryan/pointcloud_data/airport_q2/',
    "/home/bryan/pointcloud_data/evaluation_dataset/eval_20200109_075520_0",
    "/home/bryan/pointcloud_data/evaluation_dataset/eval_20200910_050202_0",
    "/home/bryan/pointcloud_data/evaluation_dataset/eval_20200910_063732_0_a",
    "/home/bryan/pointcloud_data/evaluation_dataset/eval_20200910_063732_0_b",
    "/home/bryan/pointcloud_data/evaluation_dataset/eval_20250616_035346_0_jp",
]

domain_names = [
    # 'hsinchu_q2', 'airport_q2',
    "eval_20200109_075520_0",
    "eval_20200910_050202_0",
    "eval_20200910_063732_0_a",
    "eval_20200910_063732_0_b",
    "eval_20250616_035346_0_jp",
]

ignore_index = -1
names = [
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
]

data = dict(
    num_classes=12,
    ignore_index=ignore_index,
    names=names,
    # Multi-domain training with 3-scan concatenation
    train=dict(
        type="ItriDataset",
        split="train",
        data_root=data_roots,  # Pass list for multi-domain
        domain_names=domain_names,
        domain_balance=False,  # Balance samples across domains
        concat_scans=3,  # Concatenate 3 consecutive scans
        transform=[
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            # Note: Grid sampling will handle larger point clouds from concatenation
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
    # Multi-domain validation with 3-scan concatenation
    val=dict(
        type="ItriDataset",
        split="val",
        data_root=data_roots,
        domain_names=domain_names,
        domain_balance=False,  # No balancing for validation
        concat_scans=3,  # Concatenate 3 consecutive scans
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
    # Multi-domain testing with 3-scan concatenation
    test=dict(
        type="ItriDataset",
        split="test",
        data_root=data_roots,
        domain_names=domain_names,
        domain_balance=False,
        concat_scans=1,  # Concatenate 3 consecutive scans
        transform=[
            dict(type="Copy", keys_dict=dict(segment="origin_segment")),
            dict(
                type="IntensityNormalization",
                normalization_type="percentile",
                target_range=(0, 255),
                percentiles=(5, 95),
            ),
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

# Notes on 3-scan concatenation:
# 1. Each training sample now contains 3 consecutive scans transformed to global coordinates
# 2. Poses are used to align scans in global frame for proper concatenation
# 3. Instance labels distinguish points from different scans
# 4. Batch size is reduced to handle larger point clouds
# 5. The model sees more temporal/spatial context for better learning
