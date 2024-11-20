dataset_type = 'NuImageDataset'
data_root = 'data/nuimages/'

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotation/train/nuimages_v1.0-train.json',
        img_prefix=data_root + './',
        ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotation/val/nuimages_v1.0-val.json',
        img_prefix=data_root + './',
        ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotation/val/nuimages_v1.0-val.json',
        img_prefix=data_root + './',
        ))
evaluation = dict(interval=1, metric='bbox')
