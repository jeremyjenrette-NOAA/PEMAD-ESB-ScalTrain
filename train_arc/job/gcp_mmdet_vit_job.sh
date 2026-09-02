#!/bin/bash
# ==============================================================================
# GCP Workstation MMDetection ViT Training Job
# ==============================================================================
set -e

export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

# Step 1: Fix JSON paths
python config/fix_coco_paths.py

# Step 2: Train ViTDet
python -m mmdet.tools.train configs/vitdet_scallop.py \
    --work-dir output/mmdet_vitdet_2226 \
    --cfg-options auto_scale_lr.enable=True

# Step 3: Run Inference / Export Predictions to match existing CSV benchmark format
python tools/export_mmdet_to_csv.py \
    --config configs/vitdet_scallop.py \
    --checkpoint output/mmdet_vitdet_2226/best_coco_bbox_mAP_epoch_*.pth \
    --out_csv output/mmdet_vitdet_2226/autotest2226_vitdet.csv