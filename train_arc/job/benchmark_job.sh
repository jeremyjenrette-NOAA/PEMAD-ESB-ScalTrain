#!/bin/bash
#SBATCH -J benchmark
#SBATCH --account=sharkpulse
#SBATCH --partition=normal_q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/projects/sharkpulse/archived/PEMAD-ESB-ScalTrain/train_arc/log/benchmark_cpu_%j.out
#SBATCH --error=/projects/sharkpulse/archived/PEMAD-ESB-ScalTrain/train_arc/log/benchmark_cpu_%j.err

set -e
mkdir -p ./log

# ─── 1. Environment Initialization ─────────────────────────────────────────
echo "Initializing Python ecosystem environment..."
module reset 2>/dev/null || true
# Adjust to however your Anaconda/Miniconda module is named on Tinkercliffs:
# module load Anaconda3
source ~/.bashrc
conda activate scallopdet

# export PYTORCH_ALLOC_CONF=expandable_segments:True
# export CUDA_VISIBLE_DEVICES=0
# nvidia-smi
# ─── 2. Parameter Configurations ────────────────────────────────────────────
cd /projects/sharkpulse/archived/PEMAD-ESB-ScalTrain/train_arc
python config/bench_inference.py --device cpu

echo "=== Pipeline Completed Successfully ==="

