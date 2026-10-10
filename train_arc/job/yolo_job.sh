#!/bin/bash
#SBATCH -J yolo_scal
#SBATCH --account=sharkpulse
#SBATCH --partition=t4_normal_q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=135:00:00
#SBATCH --output=/projects/sharkpulse/archived/PEMAD-ESB-ScalTrain/train_arc/log/yolo_%j.out
#SBATCH --error=/projects/sharkpulse/archived/PEMAD-ESB-ScalTrain/train_arc/log/yolo_%j.err

set -e
mkdir -p ./log

# ─── 1. Environment Initialization ─────────────────────────────────────────
echo "Initializing Python ecosystem environment..."
module reset 2>/dev/null || true
# Adjust to however your Anaconda/Miniconda module is named on Tinkercliffs:
# module load Anaconda3
source ~/.bashrc
conda activate scallopdet

export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

# ─── 2. Parameter Configurations ────────────────────────────────────────────
YEAR=2226
LABEL=data
MODEL=check/yolo12n.pt
BASE_DIR="/projects/sharkpulse/archived/PEMAD-ESB-ScalTrain/train_arc/"
PROJECT_DIR="/projects/sharkpulse/archived/PEMAD-ESB-ScalTrain/train_arc/output"
YOLO_ROOT="/projects/sharkpulse/archived/PEMAD-ESB-ScalTrain/train_arc/data2226/yolo"

JOB_ID="slurm_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
export SLURM_JOB_ID="${JOB_ID}"

MODEL_TAG=$(basename "$MODEL" .pt | tr '.-' '__')
RUN_NAME="${YEAR}${LABEL}_${MODEL_TAG}_${JOB_ID}"
RUN_DIR="${PROJECT_DIR}/${RUN_NAME}"
BEST_WEIGHTS="${RUN_DIR}/weights/best.pt"

echo "--------------------------------------------------"
echo "RUN NAME    : ${RUN_NAME}"
echo "YOLO ROOT   : ${YOLO_ROOT}"
echo "TARGET MODEL: ${MODEL}"
echo "OUTPUT DIR  : ${RUN_DIR}"
echo "--------------------------------------------------"

# ─── 3. Pre-Flight File System Integrity Checks ─────────────────────────────
echo "Verifying structural validation splits..."
if [ ! -d "${YOLO_ROOT}/images/val" ] || [ ! -d "${YOLO_ROOT}/labels/val" ]; then
    echo "ERROR: Training configuration structural splits missing inside ${YOLO_ROOT}"
    exit 1
fi

N_VAL=$(find -L "${YOLO_ROOT}/images/val" -maxdepth 1 -type f | wc -l)
echo "Validated ${N_VAL} evaluation frames ready for ingest."

# Run the dataset validator from the pipeline before spending GPU time
cd "${BASE_DIR}"
python config/05_validate_dataset.py --yolo-root "${YOLO_ROOT}"

# ─── 4. Launch Model Training ───────────────────────────────────────────────
echo "=== Step 1: Launching Ultralytics YOLO Training Workflow ==="

python config/run_yolo.py \
    --year ${YEAR} \
    --data_root "${YOLO_ROOT}" \
    --model ${MODEL} \
    --label ${LABEL} \
    --epochs 150 \
    --imgsz 640 \
    --batch 16 \
    --workers 0 \
    --project_dir "${PROJECT_DIR}" \
    --run_tag "" \
    --exist_ok

echo "=== Training Phase Complete ==="

# ─── 5. Post-Training Validation & Evaluation ───────────────────────────────
if [ ! -f "${BEST_WEIGHTS}" ]; then
    echo "ERROR: Target weight matrix file not generated at ${BEST_WEIGHTS}"
    exit 1
fi

echo -e "\n=== Step 2: Running Evaluation Metric Matrices ==="
source config/write_val.sh

VAL_IMG_CSV="${RUN_DIR}/eval/val_images${YEAR}_${MODEL_TAG}.csv"
echo "Building evaluation tracking manifests..."
write_val_image_csv "${YOLO_ROOT}" "$VAL_IMG_CSV"

python config/eval_yolo_detections_hung_multi.py \
    --year ${YEAR} \
    --model_name ${MODEL_TAG} \
    --run_dir ${RUN_DIR} \
    --weights ${BEST_WEIGHTS} \
    --data_root "${YOLO_ROOT}" \
    --out_csv ${RUN_DIR}/eval/autotest${YEAR}_${MODEL_TAG}.csv \
    --gt_out_csv ${RUN_DIR}/eval/mantest${YEAR}_${MODEL_TAG}.csv \
    --out_fn_csv ${RUN_DIR}/eval/fn${YEAR}_${MODEL_TAG}.csv \
    --imgsz 640 \
    --conf 0.1 \
    --nms_iou 0.65 \
    --match_iou 0.1 \
    --max_det 600 \
    --debug_n 10 \
    --device 0 \
    --batch 4 \
    --spname scallop  # single collapsed class, matches yolo/data.yaml

echo "=== Pipeline Completed Successfully ==="

