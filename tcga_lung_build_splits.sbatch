#!/bin/bash
#SBATCH --job-name=ec10x_kirp
#SBATCH --output=tcga_lung_splits.log
#SBATCH --nodelist=compute-0-2
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00

echo "[INFO] Running tcga_lung_build_splits.py"

python /project/hnguyen2/mvu9/folder_01_ng/Labo/tools/tcga_lung_build_splits.py

echo "[INFO] Done at $(date)"


