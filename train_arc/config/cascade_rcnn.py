# A minimal MMDetection config that fine-tunes Cascade R-CNN for 1 class.
# You can expand later (augs, schedulers, multi-scale, etc.)

_base_ = [
    # Model + training recipe base
    "mmdet::cascade_rcnn/cascade-rcnn_r50_fpn_1x_coco.py"
]

# ---- DATASET ----
data_root = "data22/"
train_ann = "coco/train.json"
val_ann   = "coco/val.json"
img_dir   = "images/"

metainfo = {
    "classes": ("scallop",),
}

train_dataloader = dict(
    batch_size=2,            # start small; scale up later
    num_workers=4,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file=train_ann,
        data_prefix=dict(img=img_dir),
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file=val_ann,
        data_prefix=dict(img=img_dir),
    )
)
test_dataloader = val_dataloader

val_evaluator = dict(
    ann_file=f"{data_root}{val_ann}",
)
test_evaluator = val_evaluator

# ---- MODEL HEADS: set num_classes=1 in all stages ----
model = dict(
    roi_head=dict(
        bbox_head=[
            dict(num_classes=1),
            dict(num_classes=1),
            dict(num_classes=1),
        ]
    )
)

# ---- TRAINING ----
work_dir = "outputs/cascade_rcnn"

# start from COCO-pretrained weights
load_from = "https://download.openmmlab.com/mmdetection/v3.0/cascade_rcnn/cascade-rcnn_r50_fpn_1x_coco/cascade-rcnn_r50_fpn_1x_coco_20221129_155340-7f6a0a8b.pth"

# a small-ish run just to prove end-to-end; extend after it works
train_cfg = dict(max_epochs=12)

# LR scaling is important. For batch_size=2 it should be small.
optim_wrapper = dict(optimizer=dict(lr=0.002))

# Logging/checkpoints
default_hooks = dict(
    checkpoint=dict(type="CheckpointHook", interval=1, max_keep_ckpts=3),
)