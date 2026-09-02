#!/bin/bash
# ==============================================================================
# GCP Workstation RT-DETR (ViT) Training & Evaluation Pipeline
# ==============================================================================
set -e

export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

YEAR=2226
LABEL=data
MODEL=rtdetr-l.pt   # Large Vision Transformer detector checkpoint

PROJECT_DIR="$(pwd)/output"
YOLO_ROOT="data2226/yolo"
JOB_ID="gcp_vit_$(date +%Y%m%d_%H%M%S)"
export SLURM_JOB_ID="${JOB_ID}"

MODEL_TAG=$(basename "$MODEL" .pt | tr '.-' '__')
RUN_NAME="${YEAR}${LABEL}_${MODEL_TAG}_${JOB_ID}"
RUN_DIR="${PROJECT_DIR}/${RUN_NAME}"
BEST_WEIGHTS="${RUN_DIR}/weights/best.pt"

echo "=== Step 1: Launching RT-DETR Vision Transformer Training ==="

# Note: Batch size adjusted to 8 for VRAM management on Tesla T4 at 1024x1024
python config/run_vit.py \
    --year ${YEAR} \
    --data_root "${YOLO_ROOT}" \
    --model ${MODEL} \
    --label ${LABEL} \
    --epochs 1 \
    --imgsz 1024 \
    --batch 4 \
    --workers 0 \
    --project_dir "${PROJECT_DIR}" \
    --run_tag "" \
    --exist_ok

echo "=== Step 2: Running Hungarian Matching Evaluation ==="
source config/write_val.sh

VAL_IMG_CSV="${RUN_DIR}/eval/val_images${YEAR}_${MODEL_TAG}.csv"
write_val_image_csv "${YOLO_ROOT}" "$VAL_IMG_CSV"

# eval_yolo_detections_hung_multi.py accepts RT-DETR weights natively
python config/eval_yolo_detections_hung_multi.py \
    --year ${YEAR} \
    --model_name ${MODEL_TAG} \
    --run_dir ${RUN_DIR} \
    --weights ${BEST_WEIGHTS} \
    --data_root "${YOLO_ROOT}" \
    --out_csv ${RUN_DIR}/eval/autotest${YEAR}_${MODEL_TAG}.csv \
    --gt_out_csv ${RUN_DIR}/eval/mantest${YEAR}_${MODEL_TAG}.csv \
    --out_fn_csv ${RUN_DIR}/eval/fn${YEAR}_${MODEL_TAG}.csv \
    --imgsz 1024 \
    --conf 0.01 \
    --nms_iou 0.65 \
    --match_iou 0.1 \
    --max_det 600 \
    --debug_n 10 \
    --device 0 \
    --batch 4 \
    --spname scallop

echo "=== Transformer Pipeline Completed Successfully ==="