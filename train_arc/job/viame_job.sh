#!/bin/bash
#SBATCH -J viame_yolo_scal
#SBATCH --account=sharkpulse
#SBATCH --partition=a30_normal_q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=60:00:00
#SBATCH --output=/projects/sharkpulse/archived/PEMAD-ESB-ScalTrain/train_arc/log/viame_yolo_scal_%j.out
#SBATCH --error=/projects/sharkpulse/archived/PEMAD-ESB-ScalTrain/train_arc/log/viame_yolo_scal_%j.err

module reset
source ~/.bashrc
conda deactivate

# Path to VIAME installation
export VIAME_INSTALL="/projects/sharkpulse/archived/viame"

source ${VIAME_INSTALL}/setup_viame.sh

BASE_DIR="/projects/sharkpulse/archived/PEMAD-ESB-ScalTrain/train_arc"
TRAIN_DIR="data2022/viame_test"

# Output directory for this job
JOB_OUT_DIR="${BASE_DIR}/output/viame_${SLURM_JOB_ID}"

# Create job-specific directory
mkdir -p "${JOB_OUT_DIR}"

# Log location
echo "Starting VIAME training job ${SLURM_JOB_ID}"
echo "Output directory: ${JOB_OUT_DIR}"
echo $CUDA_VISIBLE_DEVICES
# Adjust log level
export KWIVER_DEFAULT_LOG_LEVEL=info
export PYTORCH_ALLOC_CONF=expandable_segments:True
# Move to job output directory
cd "${JOB_OUT_DIR}"
cp ${BASE_DIR}/config/train_detector_netharn_cfrnn.conf .
# Run training
viame train \
  -i ${BASE_DIR}/${TRAIN_DIR} \
  -c ${BASE_DIR}/config/train_detector_netharn_cfrnn.conf \
  --threshold 0.0