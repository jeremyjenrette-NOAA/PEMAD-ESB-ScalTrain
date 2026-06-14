#!/bin/bash
#SBATCH -J yolo_scal
#SBATCH --account=sharkpulse
#SBATCH --partition=t4_normal_q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=30:00:00
#SBATCH --output=/projects/sharkpulse/archived/PEMAD-ESB-ScalTrain/train_arc/log/yolo_%j.out
#SBATCH --error=/projects/sharkpulse/archived/PEMAD-ESB-ScalTrain/train_arc/log/yolo_%j.err

module reset
module load Miniconda3/24.7.1-0

source ~/.bashrc
cd /projects/sharkpulse/archived/PEMAD-ESB-ScalTrain/train_arc/
conda activate scallopdet
# conda activate yolov13
# module load CUDA/12.8.0
export PYTORCH_ALLOC_CONF=expandable_segments:True
############################################################
YEAR=2226
MODEL=check/yolo12n.pt
LABEL=scallop

DATA_ROOT=data
YOLO_ROOT="${DATA_ROOT}${YEAR}/yolo"
############################################################

echo "YOLO_ROOT: ${YOLO_ROOT}"
echo "Checking validation images..."
ls -lah "${YOLO_ROOT}/images/val" | head
ls -lah "${YOLO_ROOT}/labels/val" | head
# Derive model tag exactly like run_yolo.py does
MODEL_TAG=$(basename "$MODEL" .pt | tr '.-' '__')

# Construct run name identically to Python script
RUN_NAME="${YEAR}${LABEL}_${MODEL_TAG}_${SLURM_JOB_ID}"
PROJECT_DIR="output"
RUN_DIR="${PROJECT_DIR}/${RUN_NAME}"

echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "RUN_NAME: ${RUN_NAME}"
echo "RUN_DIR: ${RUN_DIR}"

echo "=== Train start ==="
srun python config/run_yolo.py \
    --year ${YEAR} \
    --data_root "${YOLO_ROOT}" \
    --model ${MODEL} \
    --label ${LABEL} \
    --epochs 120 \
    --imgsz 1024 \
    --batch 16

echo "=== Train finished ==="

BEST_WEIGHTS="${RUN_DIR}/weights/best.pt"

if [ ! -f "${BEST_WEIGHTS}" ]; then
    echo "ERROR: best.pt not found at ${BEST_WEIGHTS}"
    exit 1
fi

echo "=== Eval start ==="

source config/write_val.sh

VAL_IMG_CSV="${RUN_DIR}/eval/val_images${YEAR}_${MODEL_TAG}.csv"

write_val_image_csv "${YOLO_ROOT}" "$VAL_IMG_CSV"

if [ ! -d "${YOLO_ROOT}/images/val" ]; then
    echo "ERROR: validation image directory does not exist:"
    echo "${YOLO_ROOT}/images/val"
    find "data${YEAR}" -maxdepth 4 -type d | sort | head -n 50
    exit 1
fi

# The -L flag tells 'find' to follow symlinks, resolving them to regular files (-type f)
N_VAL=$(find -L "${YOLO_ROOT}/images/val" -maxdepth 1 -type f | wc -l)

if [ "$N_VAL" -eq 0 ]; then
    echo "ERROR: validation image directory exists but contains zero files:"
    echo "${YOLO_ROOT}/images/val"
    find -L "${YOLO_ROOT}" -maxdepth 4 -type d | sort
    exit 1
fi

echo "Found ${N_VAL} validation images."

srun python config/eval_yolo_detections_hung.py \
    --year ${YEAR} \
    --model_name ${MODEL_TAG} \
    --run_dir ${RUN_DIR} \
    --weights ${BEST_WEIGHTS} \
    --data_root "${YOLO_ROOT}" \
    --out_csv ${RUN_DIR}/eval/autotest${YEAR}_${MODEL_TAG}.csv \
    --gt_out_csv ${RUN_DIR}/eval/mantest${YEAR}_${MODEL_TAG}.csv \
    --out_fn_csv ${RUN_DIR}/eval/fn${YEAR}_${MODEL_TAG}.csv \
    --imgsz 768 \
    --conf 0.01 \
    --nms_iou 0.65 \
    --match_iou 0.1 \
    --max_det 300 \
    --debug_n 10 \
    --device 0 \
    --batch 4

echo "=== Job complete ==="
