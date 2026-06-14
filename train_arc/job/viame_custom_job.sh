#!/bin/bash
#SBATCH -J viame_scal
#SBATCH --account=sharkpulse
#SBATCH --partition=l40s_normal_q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=145:00:00
#SBATCH --output=/projects/sharkpulse/archived/PEMAD-ESB-ScalTrain/train_arc/log/viame_%j.out
#SBATCH --error=/projects/sharkpulse/archived/PEMAD-ESB-ScalTrain/train_arc/log/viame_%j.err

module reset
source ~/.bashrc
conda deactivate

export VIAME_INSTALL="/projects/sharkpulse/archived/viame"
source ${VIAME_INSTALL}/setup_viame.sh

export KWIVER_DEFAULT_LOG_LEVEL=info
export PYTORCH_ALLOC_CONF=expandable_segments:True

YEAR=2226
DATA_ROOT=data

BASE_DIR="/projects/sharkpulse/archived/PEMAD-ESB-ScalTrain/train_arc"
JOB_OUT_DIR="${BASE_DIR}/output/${YEAR}scallop_viame_${SLURM_JOB_ID}"
TRUTH_DIR="${BASE_DIR}/${DATA_ROOT}${YEAR}/viame_truth"

TRAIN_JSON="${TRUTH_DIR}/training_truth.json"
VAL_JSON="${TRUTH_DIR}/validation_truth.json"
BACKBONE="${BASE_DIR}/check/pytorch_resnext101.pth"

mkdir -p "${JOB_OUT_DIR}"
cd "${JOB_OUT_DIR}"

echo "Starting fixed-split VIAME training job ${SLURM_JOB_ID}"
echo "Output directory: ${JOB_OUT_DIR}"
echo "Train JSON: ${TRAIN_JSON}"
echo "Val JSON: ${VAL_JSON}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

${VIAME_INSTALL}/bin/python -m viame.pytorch.netharn.detect_fit \
  --name=viame-netharn-detector \
  --arch=cascade \
  --train_dataset="${TRAIN_JSON}" \
  --vali_dataset="${VAL_JSON}" \
  --workdir="${JOB_OUT_DIR}/deep_training" \
  --xpu=0,1 \
  --workers=4 \
  --normalize_inputs=True \
  --init=noop \
  --optim=sgd \
  --augmenter=complex \
  --schedule=ReduceLROnPlateau-p2-c2 \
  --ignore_first_epochs=2 \
  --max_epoch=40 \
  --patience=7 \
  --input_dims=window \
  --window_dims=640,640 \
  --window_overlap=0.20 \
  --multiscale=False \
  --batch_size=16 \
  --bstep=4 \
  --lr=0.001 \
  --timeout=1209600 \
  --sampler_backend=none \
  --backbone_init="${BACKBONE}" \
  --allow_unicode=False \
  --channels=rgb

echo "=== Train finished ==="

echo "=== Eval start ==="
echo "=== Locating VIAME prediction directory ==="

PRED_BASE="${JOB_OUT_DIR}/deep_training/fit/nice/viame-netharn-detector/eval/validation_truth"

PRED_DIR=$(find "${PRED_BASE}" -type f -name 'dets_gid_*_v2.mscoco.json' -printf '%h\n' 2>/dev/null | sort -u | head -n 1)

if [ -z "${PRED_DIR}" ]; then
    echo "ERROR: Could not locate VIAME prediction directory under ${PRED_BASE}"
    exit 1
fi

echo "Resolved PRED_DIR: ${PRED_DIR}"
ls -lah "${PRED_DIR}"

mkdir -p "${JOB_OUT_DIR}/eval"

source config/write_val.sh

VAL_IMG_CSV="${JOB_OUT_DIR}/eval/val_images${YEAR}_cas.csv"

write_val_image_csv "data${YEAR}/yolo" "$VAL_IMG_CSV"

srun python ${BASE_DIR}/config/eval_viame_detections_hung.py \
  --year ${YEAR} \
  --model_name viame_cascade \
  --pred_dir "${PRED_DIR}" \
  --gt_json "${TRUTH_DIR}/validation_truth.json" \
  --train_json "${TRUTH_DIR}/training_truth.json" \
  --out_csv "${JOB_OUT_DIR}/eval/autotest${YEAR}_viame_cascade.csv" \
  --gt_out_csv "${JOB_OUT_DIR}/eval/mantest${YEAR}_viame_cascade.csv" \
  --out_fn_csv "${JOB_OUT_DIR}/eval/fn${YEAR}_viame_cascade.csv" \
  --conf 0.01 \
  --nms_iou 0.65 \
  --match_iou 0.1 \
  --max_det 300 \
  --debug_n 10

echo "=== Cleaning output ==="

cd "${JOB_OUT_DIR}/deep_training" || exit 1

# Remove cache-like directories if present
[ -d "_cache" ] && rm -rf "_cache"
[ -d "_mru" ] && rm -rf "_mru"

# Remove convenience name directory if present
# [ -d "fit/name" ] && rm -rf "fit/name"
# [ -d "fit/runs" ] && rm -rf "fit/runs"
