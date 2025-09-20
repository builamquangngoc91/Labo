#!/bin/bash
#SBATCH --job-name=tcga_splits
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:0
#SBATCH --qos=normal
#SBATCH --hint=nomultithread
#SBATCH --mail-type=NONE
##SBATCH --partition=your_partition_here

set -euo pipefail

echo "[INFO] Job started on $(date)"
echo "[INFO] Host: $(hostname)"

# Activate conda environment if available
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate labo || true
fi

export PYTHONUNBUFFERED=1

cd /Users/tyler/Desktop/ngoc/LaBo

echo "[INFO] Running tcga_lung_build_splits.py"
python /Users/tyler/Desktop/ngoc/LaBo/tools/tcga_lung_build_splits.py

echo "[INFO] Done at $(date)"


