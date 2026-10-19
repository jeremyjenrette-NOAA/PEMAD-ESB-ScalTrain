#!/bin/bash
#SBATCH -J tierb_2026
#SBATCH --account=sharkpulse
#SBATCH --partition=t4_normal_q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/projects/sharkpulse/archived/PEMAD-ESB-ScalTrain/train_arc/log/tierb_2026_%j.out
#SBATCH --error=/projects/sharkpulse/archived/PEMAD-ESB-ScalTrain/train_arc/log/tierb_2026_%j.err
# ==============================================================================
# Tier B (leave-one-year-out) evaluation of the 2022-2024 YOLOv12n weights.
#
#   A. 2026 test set : every annotated 2026 image (primary HabCam + Polaris + Stella),
#                      lines -> square boxes, non-scallop-only images as empties.
#   B. 2022-24 pool  : 7,078 images the 2022-2024 weights never trained on
#                      (outside the 2224 Detector-training split); the Tier B GAMs are
#                      fitted on these detections, then applied to set A.
#
# Both sets are scored with config/eval_yolo_detections_hung.py (unchanged), using the
# same settings as the 2226 evaluation the R pipeline reads, so outputs drop into
# makedat.R-style processing directly.
#
# Toggle steps:  BUILD=0 to reuse built sets; RUN_2026=0 / RUN_CALPOOL=0 to skip one.
# ==============================================================================
module reset
module load Miniconda3/24.7.1-0
source ~/.bashrc
cd /projects/sharkpulse/archived/PEMAD-ESB-ScalTrain/train_arc/
conda activate scallopdet
export PYTORCH_ALLOC_CONF=expandable_segments:True
# Device: DEVICE=auto (default) uses GPU 0 when one is visible, otherwise CPU.
# Force with DEVICE=cpu or DEVICE=0. For the CPU queue use job/eval_tierb_2026_cpu.sh.
DEVICE=${DEVICE:-auto}
if [[ "$DEVICE" == "auto" ]]; then
  if python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then DEVICE=0; else DEVICE=cpu; fi
fi
if [[ "$DEVICE" != "cpu" ]]; then export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}; DEVICE=0; fi
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
# Strict error handling only after the environment is loaded: /etc/bashrc and conda's
# activate scripts reference unset variables, so -u is deliberately not used.
set -eo pipefail

# ─── Model under test (2022-2024 weights) ──────────────────────────────────────
TRAIN_YEAR=2224
LABEL=scallop
MODEL=check/yolo12n.pt
JOB=441288
MODEL_TAG=$(basename "$MODEL" .pt | tr '.-' '__')
RUN_DIR="output/2224scallop_yolo12n_slurm_598580"
WEIGHTS="${RUN_DIR}/weights/best.pt"

# ─── Inputs ────────────────────────────────────────────────────────────────────
TIERB_DIR="data2026_tierb"
ANN_CSV="${TIERB_DIR}/annotations/annotations_2026_20260909.csv"
SPECIES_CSV="${TIERB_DIR}/annotations/SpeciesList_V6.csv"
IMAGE_ROOT="/projects/sharkpulse/archived/PEMAD-ESB-ScalPred/data/2026"   # primary/, polaris/, stella/
CALPOOL_LIST="config/tierb/calpool_2224_images.csv"
CALPOOL_SRC="data2226/yolo"                 # holds every 2022-2026 annotated image + labels
CALPOOL_DIR="data2224_calpool"

# ─── Evaluator settings (match the 2226 evaluation used by the R pipeline) ─────
CONF=0.1; NMS_IOU=0.65; MATCH_IOU=0.1; MAX_DET=600
BATCH=${BATCH:-$([[ "$DEVICE" == "cpu" ]] && echo 8 || echo 4)}

# RUN_CALPOOL defaults OFF: the 598580 weights were trained on scallop2224, which holds
# ~99.5% of ALL 2022-2024 annotated images (01_filter_annotations.py / 03_build_yolo_dataset.py),
# so calpool_2224_images.csv is NOT unseen by these weights. Only enable it for weights trained
# on the older dataset_split_2224 Detector-training split (e.g. job 441288).
BUILD=${BUILD:-1}; RUN_2026=${RUN_2026:-1}; RUN_CALPOOL=${RUN_CALPOOL:-0}
OUT_BASE="${RUN_DIR}/eval_tierb"

# ─── Pre-flight checks ─────────────────────────────────────────────────────────
for f in "$WEIGHTS" "$ANN_CSV" "$SPECIES_CSV" "$CALPOOL_LIST"; do
  [[ -f "$f" ]] || { echo "ERROR: missing $f"; exit 1; }
done
[[ -d "$IMAGE_ROOT" ]] || { echo "ERROR: missing $IMAGE_ROOT"; exit 1; }

# Use the image size the weights were trained at, and show which dataset they saw.
IMGSZ=$(python -c "import yaml;a=yaml.safe_load(open('${RUN_DIR}/args.yaml'));print(a.get('imgsz',640))" 2>/dev/null || echo 640)
TRAIN_DATA=$(python -c "import yaml;a=yaml.safe_load(open('${RUN_DIR}/args.yaml'));print(a.get('data',''))" 2>/dev/null || echo "?")
echo "Weights      : ${WEIGHTS}"
echo "Trained on   : ${TRAIN_DATA}   <-- must be the 2224 split (2022-2024 only)"
echo "imgsz        : ${IMGSZ}"
echo "device       : ${DEVICE}  (batch ${BATCH})"
echo "conf/nms/iou : ${CONF} / ${NMS_IOU} / ${MATCH_IOU}   max_det ${MAX_DET}"

# ─── 1. Build evaluation sets ─────────────────────────────────────────────────
if [[ "$BUILD" == "1" ]]; then
  if [[ "$RUN_2026" == "1" ]]; then
    echo "=== Building 2026 Tier B set ==="
    python config/build_tierb_eval.py annotations \
      --ann_csv "$ANN_CSV" --species_csv "$SPECIES_CSV" \
      --image_root "$IMAGE_ROOT" --out_root "$TIERB_DIR" \
      --platforms primary,polaris,stella --dedup_iou 0.95 --point_policy drop_image
  fi
  if [[ "$RUN_CALPOOL" == "1" ]]; then
    echo "=== Building 2022-2024 calibration pool ==="
    python config/build_tierb_eval.py list \
      --image_list "$CALPOOL_LIST" --src_yolo_root "$CALPOOL_SRC" --out_root "$CALPOOL_DIR"
  fi
fi

# ─── 2. Score with the unchanged evaluator ────────────────────────────────────
run_eval () {   # $1 = set tag, $2 = YOLO root, $3 = crop width (0 = none)
  local TAG="$1" ROOT="$2" CROP="${3:-0}" OUT="${OUT_BASE}/$1"
  mkdir -p "$OUT"
  echo "=== Evaluating ${TAG}: $(find -L "${ROOT}/images/val" -type f | wc -l) images ==="
  srun python config/eval_yolo_detections_hung.py \
    --year "${TRAIN_YEAR}" --model_name "${MODEL_TAG}" \
    --run_dir "${RUN_DIR}" --weights "${WEIGHTS}" --data_root "${ROOT}" \
    --out_csv    "${OUT}/autotest_${TAG}_${MODEL_TAG}.csv" \
    --gt_out_csv "${OUT}/mantest_${TAG}_${MODEL_TAG}.csv" \
    --out_fn_csv "${OUT}/fn_${TAG}_${MODEL_TAG}.csv" \
    --imgsz "${IMGSZ}" --conf "${CONF}" --nms_iou "${NMS_IOU}" --match_iou "${MATCH_IOU}" \
    --max_det "${MAX_DET}" --batch "${BATCH}" --device "${DEVICE}" --debug_n 10 --spname scallop \
    --crop_w "${CROP}"
}

[[ "$RUN_2026"    == "1" ]] && run_eval tierb2026   "${TIERB_DIR}/yolo" "${CROP_W:-1936}"   # 2026 PNGs are stereo pairs: left frame only
[[ "$RUN_CALPOOL" == "1" ]] && run_eval calpool2224 "${CALPOOL_DIR}/yolo" 0

# Keep the manifests next to the detections for the R side.
[[ -f "${TIERB_DIR}/tierb_manifest.csv" ]] && cp "${TIERB_DIR}"/tierb_manifest.csv "${TIERB_DIR}"/tierb_gt_lines.csv "${TIERB_DIR}"/positive_classes.csv "${OUT_BASE}/tierb2026/" 2>/dev/null || true
[[ -f "${CALPOOL_DIR}/list_manifest.csv" ]] && cp "${CALPOOL_DIR}/list_manifest.csv" "${OUT_BASE}/calpool2224/" 2>/dev/null || true

echo "=== Tier B evaluation done: ${OUT_BASE} ==="
