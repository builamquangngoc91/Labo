#!/bin/bash
#SBATCH --job-name=ec10x_kttcga_lung_copy_images
#SBATCH --output=tcga_lung_copy_images.log
#SBATCH --nodelist=compute-0-2
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00


rsync -av --info=progress2 \
    --include='*/' \
    --include='*.png' --include='*.jpg' --include='*.jpeg' \
    --include='*.tif' --include='*.tiff' --include='*.bmp' --include='*.gif' \
    --exclude='*' \
    "/project/hnguyen2/mvu9/processing_datasets/processing_tcga_256/lusc/png_patches/patch_256x256_10x"/ "/project/hnguyen2/mvu9/folder_01_ng/Labo/datasets/TCGA-LUNG/images"

echo "[INFO] Done at LUSC at $(date)"

rsync -av --info=progress2 \
    --include='*/' \
    --include='*.png' --include='*.jpg' --include='*.jpeg' \
    --include='*.tif' --include='*.tiff' --include='*.bmp' --include='*.gif' \
    --exclude='*' \
    "/project/hnguyen2/mvu9/processing_datasets/processing_tcga_256/luad/png_patches/patch_256x256_10x"/ "/project/hnguyen2/mvu9/folder_01_ng/Labo/datasets/TCGA-LUNG/images"

echo "[INFO] Done at LUAD at $(date)"


