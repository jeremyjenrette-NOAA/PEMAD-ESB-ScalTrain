#!/bin/bash
#SBATCH -J yolo_scal
#SBATCH --account=sharkpulse
#SBATCH --partition=a30_normal_q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/projects/sharkpulse/archived/PEMAD-ESB-ScalTrain/train_arc/log/yolo_scal_%j.out
#SBATCH --error=/projects/sharkpulse/archived/PEMAD-ESB-ScalTrain/train_arc/log/yolo_scal_%j.err

# ─── Load modules ──────────────────────────────────────
module reset
module load Miniconda3/24.7.1-0
# module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

# export PYTHONNOUSERSITE=1
# unset PYTHONPATH
# ─── Activate Conda environment ────────────────────────
source ~/.bashrc
cd /projects/sharkpulse/archived/PEMAD-ESB-ScalTrain/train_arc/
conda activate scallopdet

# ─── Run training ───────────────────────────────────────
srun --gres=gpu:1 nvidia-smi

echo "=== Train start ==="
srun python config/run_yolo.py --year 2022 --model check/yolov9s.pt --epochs 25 --imgsz 1024 --batch 16
