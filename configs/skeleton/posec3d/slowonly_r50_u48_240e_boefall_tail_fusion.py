model = dict(
    type='Recognizer3DV2',
        backbone=dict(
        type='ResNet3dSlowFastPose3D',
        pretrained=None,
        resample_rate=8,  # tau
        speed_ratio=8,  # alpha
        channel_ratio=8,  # beta_inv
        slow_pathway=dict(
            type='resnet3d',
            depth=50,
            pretrained=None,
            in_channels=17,
            base_channels=32,
            num_stages=3,
            out_indices=(2, ),
            stage_blocks=(4, 6, 3),
            conv1_stride_s=1,
            pool1_stride_s=1,
            inflate=(0, 1, 1),
            spatial_strides=(2, 2, 2),
            temporal_strides=(1, 1, 2),
            lateral=True,
            dilations=(1, 1, 1),
            ),
        fast_pathway=dict(
            type='resnet3d',
            depth=50,
            pretrained=None,
            lateral=True,
            base_channels=8,
            num_stages=3,
            stage_blocks=(4, 6, 3),
            out_indices=(2, ),
            conv1_kernel=(5, 7, 7),
            spatial_strides=(2, 2, 2),
            temporal_strides=(1, 1, 1),
            conv1_stride_t=1,
            pool1_stride_t=1,
            dilations=(1, 1, 1),
            norm_eval=False)),
    cls_head=dict(
        type='SlowFastHead',
        in_channels=640,  # 2048+256
        num_classes=2,
        spatial_type='avg',
        dropout_ratio=0.5),
    train_cfg=dict(aux_info=['imgs_kp']),
    test_cfg=dict(average_clips='prob',
                   aux_info=['imgs_kp']))
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
dataset_type = 'PoseDataset'
ann_file_train = 'data/boefalldown/train_rawframes.pkl'
ann_file_val = 'data/boefalldown/test_rawframes3.pkl'
left_kp = [1, 3, 5, 7, 9, 11, 13, 15]
right_kp = [2, 4, 6, 8, 10, 12, 14, 16]
train_pipeline = [
    dict(type='UniformSampleFrames', clip_len=48),
    dict(type='UniformSampleTailFrames', clip_len=8,base_offset=1,persent=0.25),
    dict(type='RandomRemoveKP',max_remove_nr=2),
    dict(type='MultiPersonProcess'),
    dict(type='RawFrameDecode'),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=False),
    dict(type='RotationTransform',max_angle=20),
    dict(type='ResizeKPAndImg', scale_kp=(-1, 64),scale_img=(-1,256)),
    dict(type='RandomResizedCropKPAndImg', area_range=(0.56, 1.0)),
    dict(type='ResizeKPAndImg', scale_kp=(56, 56),scale_img=(224,224),keep_ratio=False),
    dict(type='VideoColorJitter',color_space_aug=True),
    dict(type='Cutout'),
    dict(type='Flip', flip_ratio=0.5, left_kp=left_kp, right_kp=right_kp),
    dict(
        type='GeneratePoseTarget',
        sigma=0.6,
        use_score=True,
        with_kp=True,
        with_limb=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShapePose3D', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label','imgs_kp'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label','imgs_kp'])
]
val_pipeline = [
    dict(type='UniformSampleFrames', clip_len=48, num_clips=1, test_mode=True),
    dict(type='UniformSampleFrames', clip_len=8,base_offset=1),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(-1, 64)),
    dict(type='CenterCrop', crop_size=64),
    dict(
        type='GeneratePoseTarget',
        sigma=0.6,
        use_score=True,
        with_kp=True,
        with_limb=False),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(
        type='UniformSampleFrames', clip_len=48, num_clips=1, test_mode=True),
    dict(type='UniformSampleTailFrames', clip_len=8,base_offset=1,persent=0.25),
    dict(type='MultiPersonProcess'),
    dict(type='RawFrameDecode'),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=False),
    dict(type='ResizeKPAndImg', scale_kp=(56, 56),scale_img=(224,224),keep_ratio=False),
    dict(
        type='GeneratePoseTarget',
        sigma=0.6,
        use_score=True,
        with_kp=True,
        with_limb=False,
        double=False,
        left_kp=left_kp,
        right_kp=right_kp),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShapePose3D', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label','imgs_kp'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs','imgs_kp'])
]
data = dict(
    videos_per_gpu=16,
    workers_per_gpu=4,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix='',
        filename_tmpl='img_{:05d}.jpg',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix='',
        filename_tmpl='img_{:05d}.jpg',
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix='',
        filename_tmpl='img_{:05d}.jpg',
        pipeline=test_pipeline))
# optimizer
optimizer = dict(
    type='SGD', lr=0.2, momentum=0.9,
    weight_decay=0.0003)  # this lr is used for 8 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(policy='CosineAnnealing', by_epoch=False, min_lr=0)
checkpoint_config = dict(interval=10)
workflow = [('train', 10)]
evaluation = dict(
    interval=10,
    metrics=['top_k_accuracy', 'mean_class_accuracy'],
    topk=(1, 5))
log_config = dict(
    interval=20, hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TBSummaryPoseC3D',log_dir='tblog',interval=100),
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = '/home/wj/ai/mldata/training_data/mmaction/work_dirs/posec3d/slowonly_r50_u48_240e_boefalldown_tail_fusion'
load_from = None
total_epochs = 100
resume_from = None
find_unused_parameters = False
