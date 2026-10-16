#!/bin/bash
#SBATCH -J tierb_2026_cpu
#SBATCH --account=sharkpulse
#SBATCH --partition=normal_q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=/projects/sharkpulse/archived/PEMAD-ESB-ScalTrain/train_arc/log/tierb_2026_cpu_%j.out
#SBATCH --error=/projects/sharkpulse/archived/PEMAD-ESB-ScalTrain/train_arc/log/tierb_2026_cpu_%j.err
# CPU-queue version of eval_tierb_2026.sh: same steps and settings, inference on CPU.
# Example (sets already built):  BUILD=0 RUN_CALPOOL=0 sbatch job/eval_tierb_2026_cpu.sh
export DEVICE=cpu
exec bash /projects/sharkpulse/archived/PEMAD-ESB-ScalTrain/train_arc/job/eval_tierb_2026.sh
