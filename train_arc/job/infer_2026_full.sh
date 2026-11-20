#!/usr/bin/env bash
# ==============================================================================
# Full-survey 2026 inference with the Tier B (2022-2024) YOLOv12n weights.
# Runs in Git Bash (e.g. the RStudio terminal) on the Windows GPU server.
#
#   STEP=manifest  build manifest_2026.csv.gz from the V/W inventories (once)
#   STEP=probe     sample images: sizes, read speed, colour of V vs W duplicates, GPU check
#   STEP=infer     run chunks (resumable: re-run the same command after any stop)
#   STEP=combine   merge chunk outputs into images_/detections_2026_all.csv.gz
#
# Examples
#   STEP=manifest bash job/infer_2026_full.sh
#   STEP=probe    bash job/infer_2026_full.sh
#   STEP=infer MAX_CHUNKS=1 bash job/infer_2026_full.sh       # one 10k-image chunk as a test
#   STEP=infer SURVEY_GROUPS=habcam_primary bash job/infer_2026_full.sh
#   STEP=infer bash job/infer_2026_full.sh                    # everything left
#   STEP=combine  bash job/infer_2026_full.sh
#
# Detector settings are the Tier B evaluation's (job/eval_tierb_2026.sh):
# imgsz 640, conf 0.1, NMS IoU 0.65, max_det 600, FP16, left frame = columns [0,1936).
# ==============================================================================
set -eo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="${CONFIG_DIR:-$HERE/../config}"
[[ -f "$CONFIG_DIR/infer_2026_full.py" ]] || CONFIG_DIR="$HERE"   # scripts copied side by side

PY="${PY:-python}"
STEP="${STEP:-probe}"

# ─── Paths (Git Bash style is fine; the Python side converts /x/... to X:/...) ──
OUT="${OUT:-/d/scallop2026_tierb}"                                  # local D: drive
WEIGHTS="${WEIGHTS:-/c/Users/jeremy.jenrette/PEMAD-ESB-ScalPred/models/yolov12n_2022-2024_tierB.pt}"
INV_V="${INV_V:-$OUT/img_inventory_2026_fullpaths_v.txt}"
INV_W="${INV_W:-$OUT/img_inventory_2026_fullpaths_w.txt}"
ANNOTATED_CSV="${ANNOTATED_CSV:-}"     # optional: tierb_manifest.csv (imagename column) to flag annotated images
MANIFEST="$OUT/manifest_2026.csv.gz"

# ─── Manifest options ─────────────────────────────────────────────────────────
CHUNK_SIZE="${CHUNK_SIZE:-10000}"
SCOPE="${SCOPE:-all}"                  # all = both copies of V/W duplicates; preferred = one per image
PREFER="${PREFER:-V}"                  # copy kept as 'preferred' when an image is on both drives
CORRECTED_DRIVES="${CORRECTED_DRIVES:-V}"   # assumption, checked by STEP=probe

# ─── Throughput / selection ───────────────────────────────────────────────────
DEVICE="${DEVICE:-0}"; BATCH="${BATCH:-32}"; WORKERS="${WORKERS:-8}"
CHUNKS="${CHUNKS:-}"; SURVEY_GROUPS="${SURVEY_GROUPS:-}"; MAX_CHUNKS="${MAX_CHUNKS:-0}"
SAVE_CROPS="${SAVE_CROPS:-}"           # e.g. /d/scallop2026_left to keep every left frame as PNG
EXTRA="${EXTRA:-}"                     # any other infer_2026_full.py flags, e.g. "--dry_run"

mkdir -p "$OUT/logs"
LOG="$OUT/logs/${STEP}_$(date +%Y%m%d_%H%M%S).log"
echo "Step $STEP  |  out $OUT  |  log $LOG"
"$PY" -c "import sys; print('python', sys.version.split()[0], sys.executable)"

case "$STEP" in
  manifest)
    for f in "$INV_V" "$INV_W"; do [[ -f "$f" ]] || { echo "ERROR: missing inventory $f (set INV_V / INV_W)"; exit 1; }; done
    ANN_ARGS=(); [[ -n "$ANNOTATED_CSV" ]] && ANN_ARGS=(--annotated_csv "$ANNOTATED_CSV")
    "$PY" -u "$CONFIG_DIR/build_2026_manifest.py" \
      --inv V "$INV_V" --inv W "$INV_W" --out_dir "$OUT" \
      --chunk_size "$CHUNK_SIZE" --scope "$SCOPE" --prefer "$PREFER" \
      --corrected_drives "$CORRECTED_DRIVES" "${ANN_ARGS[@]}" 2>&1 | tee "$LOG"
    ;;
  probe)
    "$PY" -u "$CONFIG_DIR/infer_2026_full.py" --mode probe --manifest "$MANIFEST" --out_dir "$OUT" \
      --weights "$WEIGHTS" --device "$DEVICE" --batch "$BATCH" --workers "$WORKERS" \
      --probe_model $EXTRA 2>&1 | tee "$LOG"
    ;;
  infer)
    [[ -f "$WEIGHTS" ]] || { echo "ERROR: weights not found: $WEIGHTS"; exit 1; }
    SEL=(); [[ -n "$CHUNKS" ]] && SEL+=(--chunks "$CHUNKS"); [[ -n "$SURVEY_GROUPS" ]] && SEL+=(--groups "$SURVEY_GROUPS")
    [[ -n "$SAVE_CROPS" ]] && SEL+=(--save_crops "$SAVE_CROPS")
    "$PY" -u "$CONFIG_DIR/infer_2026_full.py" --mode infer --manifest "$MANIFEST" --out_dir "$OUT" \
      --weights "$WEIGHTS" --device "$DEVICE" --batch "$BATCH" --workers "$WORKERS" \
      --max_chunks "$MAX_CHUNKS" "${SEL[@]}" $EXTRA 2>&1 | tee "$LOG"
    ;;
  combine)
    "$PY" -u "$CONFIG_DIR/infer_2026_full.py" --mode combine --manifest "$MANIFEST" --out_dir "$OUT" 2>&1 | tee "$LOG"
    ;;
  *) echo "Unknown STEP=$STEP (manifest | probe | infer | combine)"; exit 1 ;;
esac
