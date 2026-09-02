#!/bin/bash
# ==============================================================================
# GCP Workstation End-to-End YOLO Training & Evaluation Pipeline
# Targets: Crab Dataset (2426), single Tesla T4 GPU workflow
# ==============================================================================
# nohup ./job/gcp_yolo_job.sh > ./log/yolo_crab_$(date +%Y%m%d_%H%M%S).log 2>&1 &
# Exit instantly if any nested pipeline step throws an error code
set -e

# ─── 1. Environment & Runtime Initialization ──────────────────────────────────
echo "Initializing Python ecosystem environment..."
# source ~/.bashrc

# Active Conda environment observed on your GCP workstation
# conda activate habcam_env

# VRAM memory management adjustments for stable training execution
export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

# ─── 2. Parameter Configurations ──────────────────────────────────────────────
YEAR=2226
LABEL=data
MODEL=check/yolo12n.pt

# FIX 1: Use $(pwd) to force an absolute path. 
# This stops Ultralytics from dropping things inside 'runs/detect/'
PROJECT="2226data_yolo12n_gcp_20260826_142130"
PROJECT_DIR="$(pwd)/output/${PROJECT}"
YOLO_ROOT="data2226/yolo"

# Generate our unique timestamp
# JOB_ID="gcp_$(date +%Y%m%d_%H%M%S)"

# FIX 2: Trick the Python script! Export this variable so run_yolo.py 
# reads it via os.environ and uses our timestamp instead of "nojob"
# export SLURM_JOB_ID="${JOB_ID}"

# The rest of your path variables will now evaluate perfectly
MODEL_TAG="YOLOv12"
# RUN_NAME="${YEAR}${LABEL}_${MODEL_TAG}_${JOB_ID}"
# RUN_DIR="${PROJECT_DIR}/${RUN_NAME}"
BEST_WEIGHTS="${PROJECT_DIR}/weights/best.pt"

echo "--------------------------------------------------"
echo "RUN NAME    : ${PROJECT}"
echo "YOLO ROOT   : ${YOLO_ROOT}"
echo "TARGET MODEL: ${MODEL}"
echo "OUTPUT DIR  : ${PROJECT_DIR}"
echo "--------------------------------------------------"

# ─── 3. Pre-Flight File System Integrity Checks ────────────────────────────────
# echo "Verifying structural validation splits..."
# if [ ! -d "${YOLO_ROOT}/images/val" ] || [ ! -d "${YOLO_ROOT}/labels/val" ]; then
#     echo "ERROR: Training configuration structural splits missing inside ${YOLO_ROOT}"
#     exit 1
# fi

# # Follow symlinks (-L) to verify physical target counts
# N_VAL=$(find -L "${YOLO_ROOT}/images/val" -maxdepth 1 -type f | wc -l)
# echo "Validated ${N_VAL} evaluation frames ready for ingest."

# # ─── 4. Launch Model Training ──────────────────────────────────────────────────
# echo "=== Step 1: Launching Ultralytics YOLO Training Workflow ==="

# # Swapped out 'srun' for direct, foreground python script processing
# python config/run_yolo.py \
#     --year ${YEAR} \
#     --data_root "${YOLO_ROOT}" \
#     --model ${MODEL} \
#     --label ${LABEL} \
#     --epochs 20 \
#     --imgsz 1024 \
#     --batch 16 \
#     --workers 0 \
#     --project_dir "${PROJECT_DIR}" \
#     --run_tag "" \
#     --exist_ok

# echo "=== Training Phase Complete ==="

# ─── 5. Post-Training Validation & Evaluation ──────────────────────────────────
if [ ! -f "${BEST_WEIGHTS}" ]; then
    echo "ERROR: Target weight matrix file not generated at ${BEST_WEIGHTS}"
    exit 1
fi

echo -e "\n=== Step 2: Running Evaluation Metric Matrices ==="
source config/write_val.sh

VAL_IMG_CSV="${PROJECT_DIR}/eval/val_images${YEAR}_${MODEL_TAG}.csv"
echo "Building evaluation tracking manifests..."
write_val_image_csv "${YOLO_ROOT}" "$VAL_IMG_CSV"

# Swapped out 'srun' for the evaluation step execution
python config/eval_yolo_detections_hung_multi.py \
    --year ${YEAR} \
    --model_name ${MODEL_TAG} \
    --run_dir ${PROJECT_DIR} \
    --weights ${BEST_WEIGHTS} \
    --data_root "${YOLO_ROOT}" \
    --out_csv ${PROJECT_DIR}/eval/autotest${YEAR}_${MODEL_TAG}.csv \
    --gt_out_csv ${PROJECT_DIR}/eval/mantest${YEAR}_${MODEL_TAG}.csv \
    --out_fn_csv ${PROJECT_DIR}/eval/fn${YEAR}_${MODEL_TAG}.csv \
    --imgsz 1024 \
    --conf 0.01 \
    --nms_iou 0.65 \
    --match_iou 0.1 \
    --max_det 600 \
    --debug_n 10 \
    --device 0 \
    --batch 4 \
    --spname scallop  # Pass all classes space-separated

echo "=== Pipeline Completed Successfully ==="