_base_ = [
    '../../_base_/models/tsm_r50.py', '../../_base_/schedules/sgd_tsm_50e.py',
    '../../_base_/default_runtime.py'
]

# model settings
model = dict(
    backbone=dict(num_segments=16),
    cls_head=dict(num_classes=3, num_segments=16))

# dataset settings
dataset_type = 'RawframeDatasetWithBBoxes'
data_root = '/home/wj/ai/smldata_0/boeoffice'
data_root_val = '/home/wj/ai/smldata_0/boeoffice'
split = 1  # official train/test splits. valid numbers: 1, 2, 3
ann_file_train = f'/home/wj/ai/smldata_0/boeoffice/train_rawframes.txt'
ann_file_val = f'/home/wj/ai/smldata_0/boeoffice/test_rawframes.txt'
ann_file_test = f'/home/wj/ai/smldata_0/boeoffice/test_rawframes.txt'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)

train_pipeline = [
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=16),
    dict(type='RawFrameDecode'),
    dict(type='RandomCropAroundBBoxes',size=900,random_crop=False),
    dict(type='Resize', scale=(-1, 256)),
    dict(
        type='MultiScaleCrop',
        input_size=224,
        scales=(1.0, 0.96875, 0.9375, 0.90625, 0.875),
        random_crop=False,
        max_wh_scale_gap=2,
        num_fixed_crops=13),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip'),
    dict(type='VideoColorJitter',color_space_aug=True),
    dict(type='Cutout'),
    dict(type='RotationTransform',max_angle=20),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=16,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='RandomCropAroundBBoxes',size=900,random_crop=False),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=16,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='RandomCropAroundBBoxes',size=900,random_crop=False),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=10,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        pipeline=test_pipeline))
evaluation = dict(
    interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy'])

# optimizer
optimizer = dict(
    lr=0.00075,  # this lr is used for 8 gpus
)
# learning policy
lr_config = dict(policy='step', step=[10, 20])
total_epochs = 20

load_from = "weights/tsm_r50_256p_1x1x8_50e_kinetics400_rgb_20200726-020785e2.pth"
# runtime settings
work_dir = './work_dirs/tsm_k400_pretrained_r50_1x1x16_boeoffice/'
