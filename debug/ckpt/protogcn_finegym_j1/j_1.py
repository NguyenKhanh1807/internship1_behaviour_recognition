modality = 'j'
graph = 'coco_new'
work_dir = './work_dirs/test_prototype/finegym/j_1'
model = dict(
    type='RecognizerGCN_7_1_1',
    backbone=dict(
        type='GCN_7_1_1',
        tcn_ms_cfg=[(3, 1), (3, 2), (3, 3), (3, 4), ('max', 3), '1x1'],
        graph_cfg=dict(
            layout='coco_new',
            mode='random',
            num_filter=8,
            init_off=0.04,
            init_std=0.02)),
    cls_head=dict(type='SimpleHead_7_4_13', num_classes=99, in_channels=384))
dataset_type = 'PoseDataset'
ann_file = '/data/lhd/pyskl_data/gym/gym_hrnet.pkl'
left_kp = [1, 3, 5, 7, 9, 11, 13, 15]
right_kp = [2, 4, 6, 8, 10, 12, 14, 16]
train_pipeline = [
    dict(type='UniformSampleFrames', clip_len=100),
    dict(type='PoseDecode'),
    dict(
        type='Flip',
        flip_ratio=0.5,
        left_kp=[1, 3, 5, 7, 9, 11, 13, 15],
        right_kp=[2, 4, 6, 8, 10, 12, 14, 16]),
    dict(type='Kinetics_Transform'),
    dict(type='GenSkeFeat', dataset='coco_new', feats=['j']),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
val_pipeline = [
    dict(type='UniformSampleFrames', clip_len=100, num_clips=1),
    dict(type='PoseDecode'),
    dict(type='Kinetics_Transform'),
    dict(type='GenSkeFeat', dataset='coco_new', feats=['j']),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
test_pipeline = [
    dict(type='UniformSampleFrames', clip_len=100, num_clips=10),
    dict(type='PoseDecode'),
    dict(type='Kinetics_Transform'),
    dict(type='GenSkeFeat', dataset='coco_new', feats=['j']),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
data = dict(
    videos_per_gpu=16,
    workers_per_gpu=4,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type='PoseDataset',
        ann_file='/data/lhd/pyskl_data/gym/gym_hrnet.pkl',
        pipeline=[
            dict(type='UniformSampleFrames', clip_len=100),
            dict(type='PoseDecode'),
            dict(
                type='Flip',
                flip_ratio=0.5,
                left_kp=[1, 3, 5, 7, 9, 11, 13, 15],
                right_kp=[2, 4, 6, 8, 10, 12, 14, 16]),
            dict(type='Kinetics_Transform'),
            dict(type='GenSkeFeat', dataset='coco_new', feats=['j']),
            dict(type='FormatGCNInput', num_person=2),
            dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['keypoint'])
        ],
        split='train'),
    val=dict(
        type='PoseDataset',
        ann_file='/data/lhd/pyskl_data/gym/gym_hrnet.pkl',
        pipeline=[
            dict(type='UniformSampleFrames', clip_len=100, num_clips=1),
            dict(type='PoseDecode'),
            dict(type='Kinetics_Transform'),
            dict(type='GenSkeFeat', dataset='coco_new', feats=['j']),
            dict(type='FormatGCNInput', num_person=2),
            dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['keypoint'])
        ],
        split='val'),
    test=dict(
        type='PoseDataset',
        ann_file='/data/lhd/pyskl_data/gym/gym_hrnet.pkl',
        pipeline=[
            dict(type='UniformSampleFrames', clip_len=100, num_clips=10),
            dict(type='PoseDecode'),
            dict(type='Kinetics_Transform'),
            dict(type='GenSkeFeat', dataset='coco_new', feats=['j']),
            dict(type='FormatGCNInput', num_person=2),
            dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['keypoint'])
        ],
        split='val'))
optimizer = dict(
    type='SGD', lr=0.025, momentum=0.9, weight_decay=0.0005, nesterov=True)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='CosineAnnealing', min_lr=0, by_epoch=False)
total_epochs = 150
checkpoint_config = dict(interval=1)
evaluation = dict(
    interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy'], topk=(1, 5))
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
gpu_ids = range(0, 1)
