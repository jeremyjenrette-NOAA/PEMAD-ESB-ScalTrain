#!/bin/bash
# ==============================================================================
# GCP Workstation End-to-End VIAME Training & Evaluation Pipeline
# Targets: Crab Dataset (2426), single Tesla T4 GPU workflow
# ==============================================================================
# nohup ./job/gcp_viame_job.sh > ./log/viame_crab_$(date +%Y%m%d_%H%M%S).log 2>&1 &
# Exit instantly if any nested pipeline step throws an error code
set -e

# ─── 1. Environment & VIAME Ecosystem Initialization ──────────────────────────
echo "Initializing Python and VIAME ecosystem paths..."
# source ~/.bashrc

# Deactivate standard conda to ensure VIAME's internal python env takes wheel precedence
# conda deactivate || true

# Source your local VIAME installation pathways
export VIAME_INSTALL="/home/user/viame"
if [ -f "${VIAME_INSTALL}/setup_viame.sh" ]; then
    source ${VIAME_INSTALL}/setup_viame.sh
else
    echo "ERROR: VIAME setup script not found at ${VIAME_INSTALL}/setup_viame.sh"
    exit 1
fi

export KWIVER_DEFAULT_LOG_LEVEL=info
export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

# ─── 2. Parameter & Directory Configurations ──────────────────────────────────
YEAR=2426
LABEL=crab
DATA_ROOT="crabdata2426"

BASE_DIR="/home/user/PEMAD-ESB-ScalTrain/train_arc"
JOB_ID="gcp_20260707_184753" 
JOB_OUT_DIR="${BASE_DIR}/output/${YEAR}${LABEL}_viame_${JOB_ID}"

# Location of your input MSCOCO formatting truth tracking JSON files
TRUTH_DIR="${BASE_DIR}/${DATA_ROOT}/viame_truth"
TRAIN_JSON="${TRUTH_DIR}/training_truth.json"
VAL_JSON="${TRUTH_DIR}/validation_truth.json"
# ─── 4. Dynamic Run Evaluation ────────────────────────────────────────────────
echo -e "\n=== Step 2: Resolving Prediction Output Environs ==="

PRED_BASE="${JOB_OUT_DIR}/deep_training/fit/nice/viame-netharn-detector/eval/validation_truth"
PRED_DIR=$(find "${PRED_BASE}" -type f -name 'dets_gid_*_v2.mscoco.json' -printf '%h\n' 2>/dev/null | sort -u | head -n 1)

if [ -z "${PRED_DIR}" ]; then
    echo "ERROR: Could not locate generated VIAME MSCOCO validation outputs under ${PRED_BASE}"
    exit 1
fi

echo "Found validation output matrix directory: ${PRED_DIR}"
mkdir -p "${JOB_OUT_DIR}/eval"

# Re-activate standard workspace environment to run Python pandas scripts
# source ~/.bashrc
# conda activate habcam_env

source ${BASE_DIR}/config/write_val.sh
VAL_IMG_CSV="${JOB_OUT_DIR}/eval/val_images${YEAR}_cas.csv"
write_val_image_csv "${BASE_DIR}/${DATA_ROOT}/yolo" "$VAL_IMG_CSV"

echo "Executing Hungarian performance evaluations..."
# Stripped out 'srun' reference execution constraints
python ${BASE_DIR}/config/eval_viame_detections_hung.py \
  --year ${YEAR} \
  --model_name viame_cascade \
  --pred_dir "${PRED_DIR}" \
  --gt_json "${VAL_JSON}" \
  --train_json "${TRAIN_JSON}" \
  --out_csv "${JOB_OUT_DIR}/eval/autotest${YEAR}_viame_cascade.csv" \
  --gt_out_csv "${JOB_OUT_DIR}/eval/mantest${YEAR}_viame_cascade.csv" \
  --out_fn_csv "${JOB_OUT_DIR}/eval/fn${YEAR}_viame_cascade.csv" \
  --conf 0.01 \
  --nms_iou 0.65 \
  --match_iou 0.1 \
  --max_det 300 \
  --debug_n 10 \
  --spname "${LABEL}"

# ─── 5. Storage Optimization Post-Clean ───────────────────────────────────────
echo -e "\n=== Step 3: Purging Temporary Cache Files ==="
cd "${JOB_OUT_DIR}/deep_training" || exit 1
[ -d "_cache" ] && rm -rf "_cache"
[ -d "_mru" ] && rm -rf "_mru"

echo "=== VIAME End-to-End GCP Job Successfully Dispatched ==="