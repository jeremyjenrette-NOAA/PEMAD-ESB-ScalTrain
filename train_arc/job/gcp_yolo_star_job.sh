#!/bin/bash
# ==============================================================================
# GCP Workstation End-to-End Pipeline: data prep -> Stage 1 train -> Stage 1
# eval -> crop extraction -> Stage 2 train -> two-stage cascade eval.
#
# This replaces the old 5-manual-step + job-script workflow with one command.
# Step 0 validates the taxonomy JSON against the ORIGINAL multi-class
# yolo/data.yaml before anything else runs — if that check fails, you find
# out in seconds instead of after a full training run with contaminated
# labels.
#
# nohup ./job/gcp_yolo_star_job.sh > ./log/yolo_$(date +%Y%m%d_%H%M%S).log 2>&1 &
# ==============================================================================
set -e

export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

YEAR=24
LABEL=star
MODEL=check/yolo12n.pt
BROAD_CLASS_NAME=star   # single-class name Stage 1 trains on

PROJECT_DIR="$(pwd)/output"
DATA_DIR="${LABEL}${YEAR}"
ORIGINAL_YOLO_ROOT="${DATA_DIR}/yolo"
BROAD_YOLO_ROOT="${DATA_DIR}/yolo_broad"
CROP_DIR="${DATA_DIR}/crops"
TAXONOMY_JSON="config/${LABEL}_taxonomy.json"
CSV_SPLIT="${DATA_DIR}/annotations/dataset_split_crab_multiclass.csv"

JOB_ID="gcp_$(date +%Y%m%d_%H%M%S)"
export SLURM_JOB_ID="${JOB_ID}"

MODEL_TAG=$(basename "$MODEL" .pt | tr '.-' '__')
RUN_NAME="${YEAR}${LABEL}_${MODEL_TAG}_${JOB_ID}"
RUN_DIR="${PROJECT_DIR}/${RUN_NAME}"
BEST_WEIGHTS="${RUN_DIR}/weights/best.pt"

echo "--------------------------------------------------"
echo "RUN NAME          : ${RUN_NAME}"
echo "ORIGINAL YOLO ROOT: ${ORIGINAL_YOLO_ROOT}"
echo "BROAD YOLO ROOT   : ${BROAD_YOLO_ROOT}"
echo "OUTPUT DIR        : ${RUN_DIR}"
echo "--------------------------------------------------"

# ─── Step 0: Validate taxonomy JSON against the ORIGINAL multi-class data.yaml ──
python -c "
from config.taxonomy_utils import load_taxonomy, validate_taxonomy_against_yolo
taxonomy = load_taxonomy('${TAXONOMY_JSON}')
validate_taxonomy_against_yolo(taxonomy, '${ORIGINAL_YOLO_ROOT}/data.yaml', taxonomy_json_path='${TAXONOMY_JSON}')
print('Taxonomy OK: ${TAXONOMY_JSON} matches ${ORIGINAL_YOLO_ROOT}/data.yaml')
"

# ─── Step 1: Remap YOLO labels for Stage 1 Broad Detection ────────────────────
# python config/prepare_broad_yolo.py \
#     --src_yolo "${ORIGINAL_YOLO_ROOT}" \
#     --dst_yolo "${BROAD_YOLO_ROOT}"

# ─── Step 2: Extract Stage 2 crops from the ORIGINAL multi-class dataset ──────
# python config/extract_crops.py \
#     --csv_split_path "${CSV_SPLIT}" \
#     --yolo_root "${ORIGINAL_YOLO_ROOT}" \
#     --output_dir "${CROP_DIR}" \
#     --taxonomy_json "${TAXONOMY_JSON}"

# ─── Step 3: Train Stage 1 Broad Detector ─────────────────────────────────────
python config/run_yolo.py \
    --year ${YEAR} \
    --data_root "${BROAD_YOLO_ROOT}" \
    --model ${MODEL} \
    --label ${LABEL} \
    --epochs 120 \
    --imgsz 1024 \
    --batch 16 \
    --workers 0 \
    --project_dir "${PROJECT_DIR}" \
    --exist_ok

# ─── Step 4: Stage 1 Broad Evaluation (validates taxonomy again internally) ───
python config/eval_yolo_detections_hung_multi.py \
    --year ${YEAR} \
    --model_name ${MODEL_TAG} \
    --run_dir ${RUN_DIR} \
    --weights ${BEST_WEIGHTS} \
    --data_root "${BROAD_YOLO_ROOT}" \
    --gt_data_root "${ORIGINAL_YOLO_ROOT}" \
    --taxonomy_json "${TAXONOMY_JSON}" \
    --out_csv ${RUN_DIR}/eval/autotest.csv \
    --gt_out_csv ${RUN_DIR}/eval/mantest.csv \
    --out_fn_csv ${RUN_DIR}/eval/fn.csv \
    --imgsz 1024 \
    --conf 0.01 \
    --nms_iou 0.65 \
    --match_iou 0.1 \
    --max_det 30 \
    --device 0 \
    --batch 4 \
    --spname ${BROAD_CLASS_NAME}

# ─── Step 5: Train Stage 2 Classifier ─────────────────────────────────────────
python config/train_classifier.py \
    --crop_dir "${CROP_DIR}" \
    --taxonomy_json "${TAXONOMY_JSON}" \
    --backbone convnext_tiny \
    --epochs 45 \
    --num_workers 0 \
    --out_weights ${RUN_DIR}/weights/${LABEL}_tax.pt

# ─── Step 6: Two-Stage Cascade Inference & Evaluation ─────────────────────────
python config/eval_two_stage_predictions.py \
    --autotest_csv ${RUN_DIR}/eval/autotest.csv \
    --val_img_dir "${ORIGINAL_YOLO_ROOT}/images/val" \
    --stage2_weights ${RUN_DIR}/weights/${LABEL}_tax.pt \
    --taxonomy_json "${TAXONOMY_JSON}" \
    --out_csv ${RUN_DIR}/eval/autotest_two_stage_cascade.csv

echo "=== Pipeline Completed Successfully ==="
