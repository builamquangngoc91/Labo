#!/bin/bash
#SBATCH --job-name=ec10x_kirp
#SBATCH --output=tcga_lung_splits.log
#SBATCH --nodelist=compute-0-2
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00

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


