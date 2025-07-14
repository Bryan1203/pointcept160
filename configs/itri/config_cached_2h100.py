_base_ = ["../_base_/default_runtime.py"]

weight = None
resume = False
evaluate = True
test_only = False
seed = 28024989
save_path = 'exp/itri/simple_cached_2h100_2025'

# Optimized for 2 H100s with 94GB VRAM each
num_worker = 16  # 8 workers per GPU
batch_size = 24  # 12 samples per GPU
batch_size_val = 16  # 8 per GPU for validation
batch_size_test = 8   # 4 per GPU for testing

epoch = 60
eval_epoch = 60
sync_bn = True
enable_amp = True

empty_cache = False
find_unused_parameters = False
mix_prob = 0.8
param_dicts = [dict(keyword='block', lr=0.002)]

hooks = [
    dict(type="CheckpointLoader"),
    dict(type='IterationTimer', warmup_iter=2),
    dict(type='InformationWriter'),
    dict(type='SemSegEvaluator'),
    dict(type='CheckpointSaver', save_freq=None),
    dict(type='PreciseEvaluator', test_last=False)
]

train = dict(type='DefaultTrainer')
test = dict(type='SemSegTester', verbose=True)

# Model configuration (unchanged)
model = dict(
    type='DefaultSegmentorV2',
    num_classes=12,
    backbone_out_channels=64,
    backbone=dict(
        type='PT-v3m1',
        in_channels=4,
        order=['z', 'z-trans', 'hilbert', 'hilbert-trans'],
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
        pdnorm_conditions=('nuScenes', 'SemanticKITTI', 'Waymo')),
    criteria=[
        dict(type='CrossEntropyLoss', loss_weight=1.0, ignore_index=-1),
        dict(type='LovaszLoss', mode='multiclass', loss_weight=1.0, ignore_index=-1)
    ])

# Adjusted learning rate for larger batch (24 vs original 12)
base_lr = 0.004  # 0.002 * (24/12) = 0.004
optimizer = dict(type='AdamW', lr=base_lr, weight_decay=0.005)

scheduler = dict(
    type='OneCycleLR',
    max_lr=[base_lr, base_lr * 0.1],
    pct_start=0.04,
    anneal_strategy='cos',
    div_factor=10.0,
    final_div_factor=100.0)

# Multi-domain data configuration
data_roots = [
    '/home/itri464058/pointcloud_data/hsinchu_q2/',
    '/home/itri464058/pointcloud_data/airport_q2/'
]
domain_names = ['hsinchu_q2', 'airport_q2']
ignore_index = -1
names = ['none', 'solid', 'broken', 'solid_solid', 'solid_broken', 'broken_solid', 
         'broken_broken', 'botts_dots', 'grass', 'curb', 'custom', 'edge']

data = dict(
    num_classes=12,
    ignore_index=-1,
    names=names,
    
    # Simple cached training data
    train=dict(
        type='FixedCachedItriDataset',  # Use the fixed version
        split='train',
        data_root=data_roots,
        domain_names=domain_names,
        domain_balance=True,
        
        # Simple cache configuration
        enable_cache=True,
        max_cache_gb=75,  # Use 75GB for cache per GPU
        cache_data_types=['coord', 'strength', 'segment'],
        cache_raw_data=True,  # Cache as numpy arrays (safer for transforms)
        
        # Full transform pipeline (caching will be handled internally)
        transform=[
            dict(type='RandomRotate', angle=[-1, 1], axis='z', center=[0, 0, 0], p=0.5),
            dict(type='RandomScale', scale=[0.9, 1.1]),
            dict(type='RandomFlip', p=0.5),
            dict(type='RandomJitter', sigma=0.005, clip=0.02),
            
            dict(type='IntensityAugmentation', 
                scale_range=(0.8, 1.2),
                shift_range=(-3.0, 3.0),
                noise_std=1.0,
                gamma_range=(0.8, 1.2),
                augment_prob=0.7),
                
            dict(type='GridSample', grid_size=0.05, hash_type='fnv', mode='train', return_grid_coord=True),
            dict(type='ToTensor'),
            dict(type='Collect', keys=('coord', 'grid_coord', 'segment'), feat_keys=('coord', 'strength'))
        ],
        
        test_mode=False,
        ignore_index=-1,
        loop=1),
    
    # Validation without caching for now (simpler)
    val=dict(
        type='ItriDataset',  # Use regular dataset for validation
        split='val',
        data_root=data_roots,
        domain_names=domain_names,
        domain_balance=False,
        transform=[
            dict(type='GridSample', grid_size=0.05, hash_type='fnv', mode='train', return_grid_coord=True),
            dict(type='ToTensor'),
            dict(type='Collect', keys=('coord', 'grid_coord', 'segment'), feat_keys=('coord', 'strength'))
        ],
        test_mode=False,
        ignore_index=-1),
    
    # Test without caching
    test=dict(
        type='ItriDataset',  # Use regular dataset for test
        split='test',
        data_root=data_roots,
        domain_names=domain_names,
        domain_balance=False,
        transform=[
            dict(type='Copy', keys_dict=dict(segment='origin_segment')),
            dict(type='GridSample', grid_size=0.05, hash_type='fnv', mode='train', return_inverse=True)
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(type='GridSample', grid_size=0.05, hash_type='fnv', mode='test', return_grid_coord=True),
            crop=None,
            post_transform=[
                dict(type='ToTensor'),
                dict(type='Collect', keys=('coord', 'grid_coord', 'index'), feat_keys=('coord', 'strength'))
            ],
            aug_transform=[
                [dict(type="RandomRotateTargetAngle", angle=[0], axis="z", center=[0, 0, 0], p=1)]
            ]),
        ignore_index=-1))

# Optional: Test without caching first
no_cache_data = dict(
    num_classes=12,
    ignore_index=-1,
    names=names,
    train=dict(
        type='ItriDataset',  # Regular dataset
        split='train',
        data_root=data_roots,
        domain_names=domain_names,
        domain_balance=True,
        transform=[
            dict(type='RandomRotate', angle=[-1, 1], axis='z', center=[0, 0, 0], p=0.5),
            dict(type='RandomScale', scale=[0.9, 1.1]),
            dict(type='RandomFlip', p=0.5),
            dict(type='RandomJitter', sigma=0.005, clip=0.02),
            dict(type='IntensityAugmentation', 
                scale_range=(0.8, 1.2), shift_range=(-3.0, 3.0),
                noise_std=1.0, gamma_range=(0.8, 1.2), augment_prob=0.7),
            dict(type='GridSample', grid_size=0.05, hash_type='fnv', mode='train', return_grid_coord=True),
            dict(type='ToTensor'),
            dict(type='Collect', keys=('coord', 'grid_coord', 'segment'), feat_keys=('coord', 'strength'))
        ],
        test_mode=False,
        ignore_index=-1,
        loop=1
    )
)