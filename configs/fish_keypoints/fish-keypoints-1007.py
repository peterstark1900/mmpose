_base_ = ['../_base_/default_runtime.py']

# runtime
train_cfg = dict(max_epochs=140, val_interval=10)

# optimizer
optim_wrapper = dict(optimizer=dict(
    type='Adam',
    lr=1e-3,
))

# learning policy
param_scheduler = [
    dict(
        type='LinearLR', begin=0, end=500, start_factor=0.001,
        by_epoch=False),  # warm-up
    dict(
        type='MultiStepLR',
        begin=0,
        end=140,
        milestones=[90, 120],
        gamma=0.1,
        by_epoch=True)
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=80)

# hooks
# default_hooks = dict(checkpoint=dict(save_best='coco/AP', rule='greater'))
default_hooks = dict(
    checkpoint=dict(save_best='EPE', rule='less', max_keep_ckpts=1))

# codec settings
codec = dict(
    type='SPR',
    input_size=(128, 128),
    heatmap_size=(32, 32),
    sigma=(4, 2),
    minimal_diagonal_length=32**0.5,
    generate_keypoint_heatmaps=True,
    decode_max_instances=30)

# model settings
model = dict(
    type='BottomupPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='HRNet',
        in_channels=3,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(32, 64)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(32, 64, 128)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(32, 64, 128, 256),
                multiscale_output=True)),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmpose/'
            'pretrain_models/hrnet_w32-36af842e.pth'),
    ),
    neck=dict(
        type='FeatureMapProcessor',
        concat=True,
    ),
    head=dict(
        type='DEKRHead',
        in_channels=480,
        num_keypoints=2,
        heatmap_loss=dict(type='KeypointMSELoss', use_target_weight=True),
        displacement_loss=dict(
            type='SoftWeightSmoothL1Loss',
            use_target_weight=True,
            supervise_empty=False,
            beta=1 / 9,
            loss_weight=0.002,
        ),
        decoder=codec,
        # This rescore net is adapted from the official repo.
        # If you are not using the original COCO dataset for training,
        # please make sure to remove the `rescore_cfg` item
        rescore_cfg=dict(
            in_channels=74,
            norm_indexes=(5, 6),
            init_cfg=dict(
                type='Pretrained',
                checkpoint='https://download.openmmlab.com/mmpose/'
                'pretrain_models/kpt_rescore_coco-33d58c5c.pth')),
    ),
    test_cfg=dict(
        multiscale_test=False,
        flip_test=True,
        nms_dist_thr=0.05,
        shift_heatmap=True,
        align_corners=False))

# enable DDP training when rescore net is used
find_unused_parameters = True


# pipelines
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='BottomupRandomAffine', input_size=codec['input_size']),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='BottomupGetHeatmapMask'),
    dict(type='PackPoseInputs'),
]
val_pipeline = [
    dict(type='LoadImage'),
    dict(
        type='BottomupResize',
        input_size=codec['input_size'],
        size_factor=1,
        resize_mode='expand'),
    dict(
        type='PackPoseInputs',
        meta_keys=('id', 'img_id', 'img_path', 'crowd_index', 'ori_shape',
                   'img_shape', 'input_size', 'input_center', 'input_scale',
                   'flip', 'flip_direction', 'flip_indices', 'raw_ann_info',
                   'skeleton_links'))
]

# base dataset settings
# dataset_type = 'ZebraDataset'
dataset_type = 'Fish0914Dataset' # 数据集类名
data_mode = 'topdown'
# data_mode = 'bottomup'
# data_root = 'data/zebra/'
data_root = 'data/Fish-Tracker-0914/'

# pipelines
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale', padding=0.8),
    dict(type='RandomFlip', direction='horizontal'),
    dict(
        type='RandomBBoxTransform',
        shift_factor=0.25,
        rotate_factor=180,
        scale_factor=(0.7, 1.3)),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]
val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale', padding=0.8),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs')
]

# data loaders
train_dataloader = dict(
    batch_size=64,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/Fish-Tracker-0914-Train.json',
        data_prefix=dict(img='images/Train/'),
        pipeline=train_pipeline,
    ))
val_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/Fish-Tracker-0914-Test.json',
        data_prefix=dict(img='images/Test/'),
        test_mode=True,
        pipeline=val_pipeline,
    ))
test_dataloader = val_dataloader

# evaluators
val_evaluator = [
    dict(type='PCKAccuracy', thr=0.2),
    dict(type='AUC'),
    dict(type='EPE'),
]
test_evaluator = val_evaluator

visualizer = dict(vis_backends=[
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
])
