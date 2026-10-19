#!/bin/bash
# Copy the Tier B (independent 2026 survey test) outputs and the Mac-built tables into the stage.
# Run from train_arc/ after copying ScalPred data/zenodo/* to ${STAGE}/ (README.md + tables).
set -eo pipefail
STAGE=${STAGE:-zenodo_stage}
SRC=${SRC:-output/2224scallop_yolo12n_slurm_598580/eval_tierb/tierb2026}
mkdir -p "${STAGE}/tierb2026"
for f in autotest_tierb2026_yolo12n.csv mantest_tierb2026_yolo12n.csv fn_tierb2026_yolo12n.csv \
         tierb_manifest.csv tierb_gt_lines.csv positive_classes.csv skipped_images.csv run_summary.csv; do
  if [[ -f "${SRC}/${f}" ]]; then cp "${SRC}/${f}" "${STAGE}/tierb2026/${f}"; echo "added tierb2026/${f}"
  else echo "note: ${SRC}/${f} not found (skipped)"; fi
done
(cd "${STAGE}/tierb2026" && md5sum *.csv > checksums_tierb2026.md5)
du -sh "${STAGE}"
