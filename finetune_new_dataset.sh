#!/bin/bash
#SBATCH --job-name=finetune_cellpt
#SBATCH --output=logs/finetune_%j.out
#SBATCH --error=logs/finetune_%j.err
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --partition=ji

# Activate environment
source ~/.bashrc
# conda activate cellpt  # Adjust if needed

# Project root
cd /hpc/group/jilab/rz179/cellpt

# --- CONFIGURATION ---
EXP_NAME="finetune_v1"
OUT_DIR="experiments/${EXP_NAME}"

# Checkpoint to load
INIT_CKPT="/hpc/group/jilab/rz179/cellpt/experiments/multiview_baselines_v3/mv_2p5x_10x_scale_entropyFlat/swin_base_patch4_window7_224_bs36_lrh0.001_20251014-100745.pt"

# Data paths (PLEASE UPDATE THESE)
# Example: TRAIN_CSV="/path/to/train.csv"
TRAIN_CSV="path/to/your/new_train.csv"
VAL_CSV="path/to/your/new_val.csv"
CLASS_MAP="path/to/your/class_map.json" 

# Model config (should match pretrained or be compatible)
MODEL="swin_base_patch4_window7_224"
IMG_SIZE=224

# Training hyperparameters
BATCH_SIZE=32
EPOCHS=50
LR_HEAD=1e-3
LR_BACKBONE=1e-5

# ---------------------

python trainer/main_trainer_finetune.py \
    --model ${MODEL} \
    --img_size ${IMG_SIZE} \
    --initial_checkpoint ${INIT_CKPT} \
    --train_csv ${TRAIN_CSV} \
    --val_csv ${VAL_CSV} \
    --class_map ${CLASS_MAP} \
    --out_dir ${OUT_DIR} \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --lr_head ${LR_HEAD} \
    --unfreeze_backbone \
    --lr_backbone ${LR_BACKBONE} \
    --num_workers 8 \
    --save_metric acc \
    --sampler uniform \
    --loss_type ce \
    --label_smoothing 0.1 \
    --grad_checkpoint

echo "Done. Results in ${OUT_DIR}"
