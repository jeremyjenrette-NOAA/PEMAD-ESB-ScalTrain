#!/bin/bash
# ==============================================================================
# GCP workstation end-to-end pipeline, dynamic across systems.
#
#   ./job/gcp_yolo_job.sh <system>        # system = a profile in job/systems/<system>.env
#
#   nohup ./job/gcp_yolo_job.sh crab  > ./log/yolo_crab_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#   nohup ./job/gcp_yolo_job.sh star  > ./log/yolo_star_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#
# Two-stage systems (crab, star):
#   Step -1 build yolo/ from raw layout (if missing) -> 0 taxonomy gate -> 1 broad labels -> 2 crops -> 3 Stage 1 train
#   -> 4 Stage 1 eval -> 5 Stage 2 train -> 6 cascade eval
# (Scallop is single-stage and keeps its own scripts: gcp_yolo_scallop_job.sh / gcp_eval_scallop.sh.)
#
# Steps -1, 1 and 2 are skipped when their outputs already exist; FORCE_PREP=1 redoes them.
# To add a new system: add job/systems/<name>.env and config/<LABEL>_taxonomy.json.
# Run from train_arc/.
# ==============================================================================
set -euo pipefail

SYSTEM="${1:?usage: $0 <system>   (available: $(ls "$(dirname "$0")"/systems/*.env | xargs -n1 basename | sed 's/\.env$//' | tr '\n' ' '))}"
PROFILE="$(dirname "$0")/systems/${SYSTEM}.env"
[ -f "${PROFILE}" ] || { echo "ERROR: no system profile ${PROFILE}"; exit 1; }
# shellcheck disable=SC1090
source "${PROFILE}"

export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

PROJECT_DIR="$(pwd)/output"
DATA_DIR="${LABEL}${YEAR}"
ORIGINAL_YOLO_ROOT="${DATA_DIR}/yolo"
BROAD_YOLO_ROOT="${DATA_DIR}/yolo_broad"
CROP_DIR="${DATA_DIR}/crops"
TAXONOMY_JSON="config/${LABEL}_taxonomy.json"
CSV_SPLIT="${DATA_DIR}/annotations/${CSV_SPLIT_NAME:-}"

STAGE1_ROOT="${BROAD_YOLO_ROOT}"

JOB_ID="gcp_$(date +%Y%m%d_%H%M%S)"
export SLURM_JOB_ID="${JOB_ID}"   # run_yolo.py reads this for the run name

MODEL_TAG=$(basename "$MODEL" .pt | tr '.-' '__')
RUN_NAME="${YEAR}${LABEL}_${MODEL_TAG}_${JOB_ID}"
RUN_DIR="${PROJECT_DIR}/${RUN_NAME}"
BEST_WEIGHTS="${RUN_DIR}/weights/best.pt"

echo "--------------------------------------------------"
echo "SYSTEM            : ${SYSTEM}"
echo "RUN NAME          : ${RUN_NAME}"
echo "ORIGINAL YOLO ROOT: ${ORIGINAL_YOLO_ROOT}"
echo "STAGE 1 ROOT      : ${STAGE1_ROOT}"
echo "OUTPUT DIR        : ${RUN_DIR}"
echo "--------------------------------------------------"

# ─── Step -1: build the multi-class YOLO set from the raw layout, if needed ───
# ${DATA_DIR}/annotations/{dataset_split*.csv,groundtruth*.csv}, ${DATA_DIR}/annotated/, ${DATA_DIR}/empties/
# Class ids come from the taxonomy JSON. Rebuilt when yolo/data.yaml is missing or FORCE_PREP=1.
if [ "${FORCE_PREP:-0}" = "1" ] || [ ! -f "${ORIGINAL_YOLO_ROOT}/data.yaml" ]; then
    [ -d "${DATA_DIR}/annotations" ] && [ -d "${DATA_DIR}/annotated" ] || {
        echo "ERROR: ${DATA_DIR}/yolo is missing and there is no raw layout (annotations/, annotated/) to build it from"; exit 1; }
    python config/build_multiclass_yolo.py \
        --base-dir "${DATA_DIR}" \
        --taxonomy_json "${TAXONOMY_JSON}"
else
    echo "Step -1 skipped: ${ORIGINAL_YOLO_ROOT}/data.yaml exists (FORCE_PREP=1 to rebuild)"
fi

[ -d "${ORIGINAL_YOLO_ROOT}/images/val" ] && [ -d "${ORIGINAL_YOLO_ROOT}/labels/val" ] || {
    echo "ERROR: ${ORIGINAL_YOLO_ROOT} is missing images/val or labels/val"; exit 1; }

# ─── Step 0: taxonomy gate ────────────────────────────────────────────────────
[ -f "${TAXONOMY_JSON}" ] || { echo "ERROR: ${TAXONOMY_JSON} not found (needed for ${SYSTEM})"; exit 1; }
python -c "
from config.taxonomy_utils import load_taxonomy, validate_taxonomy_against_yolo
t = load_taxonomy('${TAXONOMY_JSON}')
validate_taxonomy_against_yolo(t, '${ORIGINAL_YOLO_ROOT}/data.yaml', taxonomy_json_path='${TAXONOMY_JSON}')
print('Taxonomy OK: ${TAXONOMY_JSON} matches ${ORIGINAL_YOLO_ROOT}/data.yaml')
"

# ─── Step 1: broad single-class labels for Stage 1 ────────────────────────
if [ "${FORCE_PREP:-0}" = "1" ] || [ ! -f "${BROAD_YOLO_ROOT}/data.yaml" ]; then
    python config/prepare_broad_yolo.py \
        --src_yolo "${ORIGINAL_YOLO_ROOT}" \
        --dst_yolo "${BROAD_YOLO_ROOT}" \
        --broad_class_name "${BROAD_CLASS_NAME}"
else
    echo "Step 1 skipped: ${BROAD_YOLO_ROOT} exists (FORCE_PREP=1 to redo)"
fi

# ─── Step 2: Stage 2 crops from the ORIGINAL multi-class set ──────────────
if [ "${FORCE_PREP:-0}" = "1" ] || [ ! -f "${CROP_DIR}/crop_manifest.csv" ]; then
    python config/extract_crops.py \
        --csv_split_path "${CSV_SPLIT}" \
        --yolo_root "${ORIGINAL_YOLO_ROOT}" \
        --output_dir "${CROP_DIR}" \
        --taxonomy_json "${TAXONOMY_JSON}"
else
    echo "Step 2 skipped: ${CROP_DIR}/crop_manifest.csv exists (FORCE_PREP=1 to redo)"
fi

# ─── Step 3: train Stage 1 (broad) detector ───────────────────────────────────
python config/run_yolo.py \
    --year "${YEAR}" \
    --data_root "${STAGE1_ROOT}" \
    --model "${MODEL}" \
    --label "${LABEL}" \
    --epochs "${EPOCHS}" \
    --imgsz 1024 \
    --batch 16 \
    --workers 0 \
    --project_dir "${PROJECT_DIR}" \
    --run_tag "" \
    --exist_ok

[ -f "${BEST_WEIGHTS}" ] || { echo "ERROR: weights not generated at ${BEST_WEIGHTS}"; exit 1; }

source config/write_val.sh
VAL_IMG_CSV="${RUN_DIR}/eval/val_images${YEAR}_${MODEL_TAG}.csv"
write_val_image_csv "${STAGE1_ROOT}" "$VAL_IMG_CSV"

# ─── Step 4: Stage 1 evaluation (GT labels come from the ORIGINAL multi-class set, taxonomy-validated) ──
S1_OUT="${RUN_DIR}/eval/autotest.csv"

python config/eval_yolo_detections_hung_multi.py \
    --year "${YEAR}" \
    --model_name "${MODEL_TAG}" \
    --run_dir "${RUN_DIR}" \
    --weights "${BEST_WEIGHTS}" \
    --data_root "${STAGE1_ROOT}" \
    --gt_data_root "${ORIGINAL_YOLO_ROOT}" \
    --taxonomy_json "${TAXONOMY_JSON}" \
    --out_csv "${S1_OUT}" \
    --gt_out_csv "${RUN_DIR}/eval/mantest.csv" \
    --out_fn_csv "${RUN_DIR}/eval/fn.csv" \
    --imgsz 1024 \
    --conf 0.01 \
    --nms_iou 0.65 \
    --match_iou 0.1 \
    --max_det "${MAX_DET}" \
    --device 0 \
    --batch 4 \
    --spname ${BROAD_CLASS_NAME}

# ─── Step 5: train Stage 2 hierarchical classifier ────────────────────────────
python config/train_classifier.py \
    --crop_dir "${CROP_DIR}" \
    --taxonomy_json "${TAXONOMY_JSON}" \
    --backbone "${CLS_BACKBONE}" \
    --epochs "${CLS_EPOCHS}" \
    --num_workers 0 \
    --out_weights "${RUN_DIR}/weights/${LABEL}_tax.pt"

# ─── Step 6: two-stage cascade evaluation ─────────────────────────────────────
python config/eval_two_stage_predictions.py \
    --autotest_csv "${S1_OUT}" \
    --val_img_dir "${ORIGINAL_YOLO_ROOT}/images/val" \
    --stage2_weights "${RUN_DIR}/weights/${LABEL}_tax.pt" \
    --taxonomy_json "${TAXONOMY_JSON}" \
    --out_csv "${RUN_DIR}/eval/autotest_two_stage_cascade.csv"

echo "=== Pipeline Completed Successfully (${SYSTEM}) ==="
