#!/usr/bin/env bash
# ==============================================================================
# Full-survey 2026 inference with the Tier B (2022-2024) YOLOv12n weights.
# Runs in Git Bash (e.g. the RStudio terminal) on the Windows GPU server.
#
#   STEP=manifest  build manifest_2026.csv.gz from the V/W inventories (once)
#   STEP=probe     sample images: sizes, read speed, colour of V vs W duplicates, GPU check
#   STEP=infer     run chunks (resumable: re-run the same command after any stop)
#   STEP=check     score annotated images (CHECK_CSV) and compare with the Tier B eval (REF_CSV)
#   STEP=combine   merge chunk outputs into images_/detections_2026_all.csv.gz
#
# Examples
#   STEP=manifest bash job/infer_2026_full.sh
#   STEP=probe    bash job/infer_2026_full.sh
#   STEP=check    bash job/infer_2026_full.sh                 # ~1,000 annotated images vs Tier B eval
#   STEP=infer DETACH=1 bash job/infer_2026_full.sh           # headless: survives closing the terminal
#   tail -f /d/scallop2026_tierb/logs/infer_detached_*.log    # follow it
#   touch /d/scallop2026_tierb/STOP                           # stop cleanly after the current chunk
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
# PyTorch on Windows: if `python -c "import torch"` fails with WinError 1114 (c10.dll), install the
# latest Microsoft Visual C++ Redistributable x64 (https://aka.ms/vs/17/release/vc_redist.x64.exe),
# and make sure torch is a CUDA build: python -c "import torch;print(torch.__version__)" should end +cuXXX.

# ─── Paths (Git Bash style is fine; the Python side converts /x/... to X:/...) ──
OUT="${OUT:-/d/scallop2026_tierb}"                                  # local D: drive
WEIGHTS="${WEIGHTS:-/c/Users/jeremy.jenrette/PEMAD-ESB-ScalPred/models/yolov12n_2022-2024_tierB.pt}"
INV_V="${INV_V:-$OUT/img_inventory_2026_fullpaths_v.txt}"
INV_W="${INV_W:-$OUT/img_inventory_2026_fullpaths_w.txt}"
ANNOTATED_CSV="${ANNOTATED_CSV:-$OUT/tierb_manifest.csv}"           # flags (and always keeps) annotated images
CHECK_CSV="${CHECK_CSV:-$CONFIG_DIR/tierb/tierb2026_check_1000.csv}"  # annotated images for STEP=check
REF_CSV="${REF_CSV:-$OUT/autotest_tierb2026_yolo12n.csv}"           # Tier B eval detections (cluster)
MANIFEST="$OUT/manifest_2026.csv.gz"

# ─── Manifest options ─────────────────────────────────────────────────────────
CHUNK_SIZE="${CHUNK_SIZE:-10000}"
SCOPE="${SCOPE:-all}"                  # all = both copies of V/W duplicates; preferred = one per image
PREFER="${PREFER:-V}"                  # copy kept as 'preferred' when an image is on both drives
CORRECTED_DRIVES="${CORRECTED_DRIVES:-V}"   # assumption, checked by STEP=probe
FRAME_STEP="${FRAME_STEP:-5}"          # defines the every-5th-frame subset (column frame_keep)
SELECT="${SELECT:-all}"                # all = process every image; frames = only the frame_keep subset
# Files already processed (images_c*.csv in $OUT/chunks) always keep their chunk and are never redone.

# ─── Throughput / selection ───────────────────────────────────────────────────
DEVICE="${DEVICE:-0}"; BATCH="${BATCH:-32}"; WORKERS="${WORKERS:-8}"
CHUNKS="${CHUNKS:-}"; SURVEY_GROUPS="${SURVEY_GROUPS:-}"; MAX_CHUNKS="${MAX_CHUNKS:-0}"
SAVE_CROPS="${SAVE_CROPS:-}"           # off; e.g. /d/scallop2026_left to keep every left frame as PNG
EXTRA="${EXTRA:-}"                     # any other infer_2026_full.py flags, e.g. "--dry_run"

mkdir -p "$OUT/logs"

# ─── Headless: re-launch this script under nohup, detached from the terminal ──
if [[ "${DETACH:-0}" == "1" ]]; then
  DLOG="$OUT/logs/${STEP}_detached_$(date +%Y%m%d_%H%M%S).log"
  command -v nohup >/dev/null || { echo "ERROR: nohup not found in this shell"; exit 1; }
  DETACH=0 nohup bash "${BASH_SOURCE[0]}" > "$DLOG" 2>&1 < /dev/null &
  PID=$!
  disown "$PID" 2>/dev/null || true
  echo "$PID" > "$OUT/logs/${STEP}.pid"
  echo "Started STEP=$STEP in the background (bash PID $PID)."
  echo "  follow : tail -f $DLOG"
  echo "  stop   : touch $OUT/STOP      (finishes the current chunk, then exits; re-run to resume)"
  echo "  running: tasklist | grep -i python"
  exit 0
fi

LOG="$OUT/logs/${STEP}_$(date +%Y%m%d_%H%M%S).log"
echo "Step $STEP  |  out $OUT  |  log $LOG"
"$PY" -c "import sys; print('python', sys.version.split()[0], sys.executable)"

case "$STEP" in
  manifest)
    for f in "$INV_V" "$INV_W"; do [[ -f "$f" ]] || { echo "ERROR: missing inventory $f (set INV_V / INV_W)"; exit 1; }; done
    ANN_ARGS=(); [[ -f "$ANNOTATED_CSV" ]] && ANN_ARGS=(--annotated_csv "$ANNOTATED_CSV") \
      || echo "NOTE: $ANNOTATED_CSV not found; annotated images will not be flagged"
    "$PY" -u "$CONFIG_DIR/build_2026_manifest.py" \
      --inv V "$INV_V" --inv W "$INV_W" --out_dir "$OUT" \
      --chunk_size "$CHUNK_SIZE" --scope "$SCOPE" --prefer "$PREFER" \
      --corrected_drives "$CORRECTED_DRIVES" --frame_step "$FRAME_STEP" --select "$SELECT" \
      --done_chunks "$OUT/chunks" "${ANN_ARGS[@]}" 2>&1 | tee "$LOG"
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
  check)
    for f in "$CHECK_CSV" "$REF_CSV"; do [[ -f "$f" ]] || { echo "ERROR: missing $f"; exit 1; }; done
    "$PY" -u "$CONFIG_DIR/infer_2026_full.py" --mode infer --manifest "$MANIFEST" --out_dir "$OUT" \
      --weights "$WEIGHTS" --device "$DEVICE" --batch "$BATCH" --workers "$WORKERS" \
      --subset_csv "$CHECK_CSV" $EXTRA 2>&1 | tee "$LOG"
    SUB="$OUT/subset_$(basename "$CHECK_CSV" .csv)"
    "$PY" -u "$CONFIG_DIR/infer_2026_full.py" --mode compare --manifest "$MANIFEST" --out_dir "$SUB" \
      --ref_csv "$REF_CSV" 2>&1 | tee -a "$LOG"
    ;;
  combine)
    "$PY" -u "$CONFIG_DIR/infer_2026_full.py" --mode combine --manifest "$MANIFEST" --out_dir "$OUT" 2>&1 | tee "$LOG"
    ;;
  *) echo "Unknown STEP=$STEP (manifest | probe | check | infer | combine)"; exit 1 ;;
esac
