#!/bin/bash
#SBATCH -J yolo_scal
#SBATCH --account=sharkpulse
#SBATCH --partition=a30_normal_q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=/projects/sharkpulse/archived/PEMAD-ESB-ScalTrain/train_arc/log/yolo_scal_%j.out
#SBATCH --error=/projects/sharkpulse/archived/PEMAD-ESB-ScalTrain/train_arc/log/yolo_scal_%j.err

module reset
module load Miniconda3/24.7.1-0

source ~/.bashrc
cd /projects/sharkpulse/archived/PEMAD-ESB-ScalTrain/train_arc/
conda activate scallopdet
export PYTORCH_ALLOC_CONF=expandable_segments:True
############################################################
YEAR=2224
MODEL=check/yolo12n.pt
LABEL=scallop
############################################################
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
    --model ${MODEL} \
    --label ${LABEL} \
    --epochs 100 \
    --imgsz 1024 \
    --batch 16

echo "=== Train finished ==="

BEST_WEIGHTS="${RUN_DIR}/weights/best.pt"

if [ ! -f "${BEST_WEIGHTS}" ]; then
    echo "ERROR: best.pt not found at ${BEST_WEIGHTS}"
    exit 1
fi

echo "=== Eval start ==="

srun python config/eval_yolo_detections_hung.py \
    --year ${YEAR} \
    --model_name ${MODEL_TAG} \
    --run_dir ${RUN_DIR} \
    --weights ${BEST_WEIGHTS} \
    --data_root data${YEAR}/yolo \
    --out_csv ${RUN_DIR}/eval/autotest${YEAR}_${MODEL_TAG}.csv \
    --gt_out_csv ${RUN_DIR}/eval/mantest${YEAR}_${MODEL_TAG}.csv \
    --out_fn_csv ${RUN_DIR}/eval/fn${YEAR}_${MODEL_TAG}.csv \
    --imgsz 768 \
    --conf 0.01 \
    --nms_iou 0.65 \
    --match_iou 0.1 \
    --max_det 150 \
    --debug_n 10 \
    --device 0 \
    --batch 1

echo "=== Job complete ==="
