#!/bin/bash
#SBATCH -J scal_train22_test
#SBATCH --account=sharkpulse
#SBATCH --partition=a100_normal_q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=22:30:00
#SBATCH --output=/projects/sharkpulse/archived/PEMAD-ESB-ScalTrain/train_arc/log/scal_train22_test_%j.out
#SBATCH --output=/projects/sharkpulse/archived/PEMAD-ESB-ScalTrain/train_arc/log/scal_train22_test_%j.err

module load Miniconda3/24.7.1-0
module load PyTorch/2.1.2-foss-2023a

module load cuda/12.1  # adjust to your cluster
source ~/.bashrc
conda activate scallopdet

# optional: speed up dataloading
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# train
mim train mmdet configs/cascade_rcnn_scallop.py