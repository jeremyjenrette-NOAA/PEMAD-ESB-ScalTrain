#!/bin/bash
# Copy the Tier B (independent 2026 survey test) files into the Zenodo stage.
# Run from train_arc/ AFTER the cropped (left-frame) eval_tierb_2026 rerun.
#   STAGE     stage root (default zenodo_stage)
#   SRC       eval output dir (default the 598580 tierb2026 folder)
#   SPLIT_CSV dataset_split_tierb2026.csv made on the Mac by s/tierb/run_tierb.sh
#             (copy it to ARC first; skipped with a warning if absent)
set -eo pipefail
STAGE=${STAGE:-zenodo_stage}
SRC=${SRC:-output/2224scallop_yolo12n_slurm_598580/eval_tierb/tierb2026}
SPLIT_CSV=${SPLIT_CSV:-}
OUT="${STAGE}/tierb2026"

# Guard against re-staging the uncropped run: the fixed manifest has eval_w = 1936.
python - "$SRC/tierb_manifest.csv" <<'PY'
import sys, pandas as pd
m = pd.read_csv(sys.argv[1])
if "eval_w" not in m or m["eval_w"].dropna().max() > 1936:
    sys.exit("tierb_manifest.csv is from the uncropped (stereo-pair) run; rerun eval_tierb_2026.sh first.")
print("manifest OK: left-frame evaluation (eval_w = 1936)")
PY

rm -rf "$OUT"; mkdir -p "$OUT"
for f in autotest_tierb2026_yolo12n.csv mantest_tierb2026_yolo12n.csv fn_tierb2026_yolo12n.csv \
         tierb_gt_lines.csv positive_classes.csv skipped_images.csv run_summary.csv; do
  if [[ -f "${SRC}/${f}" ]]; then cp "${SRC}/${f}" "${OUT}/${f}"; echo "added tierb2026/${f}"
  else echo "note: ${SRC}/${f} not found (skipped)"; fi
done
# Manifest without cluster file paths
python - "$SRC/tierb_manifest.csv" "$OUT/tierb_manifest.csv" <<'PY'
import sys, pandas as pd
pd.read_csv(sys.argv[1]).drop(columns=["img_path"], errors="ignore").to_csv(sys.argv[2], index=False)
PY
echo "added tierb2026/tierb_manifest.csv (img_path dropped)"
if [[ -n "$SPLIT_CSV" && -f "$SPLIT_CSV" ]]; then
  cp "$SPLIT_CSV" "${OUT}/dataset_split_tierb2026.csv"; echo "added tierb2026/dataset_split_tierb2026.csv"
else
  echo "WARNING: SPLIT_CSV not set/found - Calibration-fitting/Holdout assignments NOT staged"
fi
(cd "$OUT" && md5sum *.csv > checksums_tierb2026.md5)
ls -la "$OUT"; du -sh "${STAGE}"
