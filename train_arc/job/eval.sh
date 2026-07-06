#!/bin/bash
#SBATCH -J eval_yolo
#SBATCH --account=sharkpulse
#SBATCH --partition=t4_normal_q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=20:00:00
#SBATCH --output=/projects/sharkpulse/archived/PEMAD-ESB-ScalTrain/train_arc/log/eval_yolo_%j.out
#SBATCH --error=/projects/sharkpulse/archived/PEMAD-ESB-ScalTrain/train_arc/log/eval_yolo_%j.err

# ─── Load modules ──────────────────────────────────────
module reset
module load Miniconda3/24.7.1-0
# ─── Activate Conda environment ────────────────────────
source ~/.bashrc
cd /projects/sharkpulse/archived/PEMAD-ESB-ScalTrain/train_arc/
conda activate scallopdet
export PYTORCH_ALLOC_CONF=expandable_segments:True
# ─── Run training ───────────────────────────────────────
YEAR=2226
MODEL=check/yolo12n.pt
JOB=441288
LABEL=scallop
export CUDA_VISIBLE_DEVICES=0
# ────────────────────────────────────────────────────────
# Derive model tag exactly like run_yolo.py does
MODEL_TAG=$(basename "$MODEL" .pt | tr '.-' '__')

# Construct run name identically to Python script
RUN_NAME="${YEAR}${LABEL}_${MODEL_TAG}_${JOB}"
PROJECT_DIR="output"
RUN_DIR="${PROJECT_DIR}/${RUN_NAME}"
BEST_WEIGHTS="${RUN_DIR}/weights/best.pt"

echo "=== Evaluation start ==="

source config/write_val.sh

VAL_IMG_CSV="${RUN_DIR}/eval/val_images${YEAR}_${MODEL_TAG}.csv"

write_val_image_csv "data${YEAR}/yolo" "$VAL_IMG_CSV"

srun python config/eval_yolo_detections_hung.py \
    --year ${YEAR} \
    --model_name ${MODEL_TAG} \
    --run_dir ${RUN_DIR} \
    --weights ${BEST_WEIGHTS} \
    --data_root data${YEAR}/yolo \
    --out_csv ${RUN_DIR}/eval/autotest${YEAR}_${MODEL_TAG}.csv \
    --gt_out_csv ${RUN_DIR}/eval/mantest${YEAR}_${MODEL_TAG}.csv \
    --out_fn_csv ${RUN_DIR}/eval/fn${YEAR}_${MODEL_TAG}.csv \
    --imgsz 1024 \
    --conf 0.01 \
    --nms_iou 0.65 \
    --match_iou 0.1 \
    --max_det 300 \
    --debug_n 10 \
    --device 0 \
    --batch 4
