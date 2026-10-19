#!/bin/bash
#SBATCH -J stage_zenodo
#SBATCH --account=sharkpulse
#SBATCH --partition=normal_q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/projects/sharkpulse/archived/PEMAD-ESB-ScalTrain/train_arc/log/stage_zenodo_%j.out
#SBATCH --error=/projects/sharkpulse/archived/PEMAD-ESB-ScalTrain/train_arc/log/stage_zenodo_%j.err
# ==============================================================================
# Stage the cluster-side Zenodo files: Calibration-set image archives + model weights.
# Tabular files are built on the Mac by PEMAD-ESB-ScalPred/s/build_zenodo_tables.py;
# copy its image_metadata.csv to ${STAGE}/ before running this job.
# Weights that are missing are skipped with a warning (set the paths below).
# ==============================================================================
module reset
module load Miniconda3/24.7.1-0
source ~/.bashrc
cd /projects/sharkpulse/archived/PEMAD-ESB-ScalTrain/train_arc/
conda activate scallopdet
set -eo pipefail

STAGE="zenodo_stage"
SRC_IMAGES="data2226/yolo/images/val"          # = Calibration set (symlinks to originals)
mkdir -p "${STAGE}/images" "${STAGE}/weights"

# ─── Weights (edit paths if needed) ──────────────────────────────────────────
declare -A WEIGHTS=(
  [yolov12n_2022-2026.pt]="output/2226data_yolo12n_slurm_598585/weights/best.pt"
  [yolov12n_2022-2024_tierB.pt]="output/2224scallop_yolo12n_slurm_598580/weights/best.pt"
  [rtdetr_l_2022-2026.pt]="${RTDETR_WEIGHTS:-output/2226data_rtdetr_l_gcp_vit_20260924_190833/weights/best.pt}"
)
for name in "${!WEIGHTS[@]}"; do
  src="${WEIGHTS[$name]}"
  if [[ -f "$src" ]]; then cp -L "$src" "${STAGE}/weights/${name}"; echo "weights: ${name} <- ${src}"
  else echo "WARNING: missing weights ${src} (skipped)"; fi
done
# Cascade R-CNN: VIAME/netharn deploy package (deploy_snapshot.pt + MM_CascadeRCNN model code +
# train_info.json), exported at the end of training. No retraining needed.
CASCADE_DEPLOY="${CASCADE_DEPLOY:-output/2226data_viame_slurm_598695/deploy.zip}"
if [[ -f "$CASCADE_DEPLOY" ]]; then
  cp -L "$CASCADE_DEPLOY" "${STAGE}/weights/cascade_rcnn_viame_deploy_2022-2026.zip"
  echo "weights: cascade_rcnn_viame_deploy_2022-2026.zip <- ${CASCADE_DEPLOY}"
else echo "WARNING: missing Cascade deploy package ${CASCADE_DEPLOY} (skipped)"; fi
(cd "${STAGE}/weights" && md5sum * > checksums_weights.md5 || true)

# ─── Images ──────────────────────────────────────────────────────────────────
python config/package_zenodo_images.py \
  --image_metadata "${STAGE}/image_metadata.csv" \
  --src_dir "${SRC_IMAGES}" --out_dir "${STAGE}/images" --max_gb "${MAX_GB:-2}"

du -sh "${STAGE}"/* ; echo "=== Zenodo staging done: ${STAGE} ==="
