#!/bin/bash
# run_all.sh — 顺序运行 6 个比例消融实验 (S2 复用 baseline)
#
# Usage:
#   bash 0514/scripts/ratio_ablation/run_all.sh [num_epochs]
#
# 默认 num_epochs=20

set -e
cd /data1/ouyangxinglong/GBP-Cascade

NUM_EPOCHS=${1:-20}
BASE_VAL="/data1/ouyangxinglong/GBP-Cascade/0514/logs/20260514_task_risk_SwinV2Tiny_ordinal_trimodal_1/val_split.xlsx"
BASE_TRAIN="/data1/ouyangxinglong/GBP-Cascade/0514/logs/20260514_task_risk_SwinV2Tiny_ordinal_trimodal_1/train_split.xlsx"
SUB_DIR="/data1/ouyangxinglong/GBP-Cascade/0514dataset_flat/ratio_ablation_splits"

# Step 0: 准备子集（若已存在则跳过）
if [ ! -f "$SUB_DIR/train_D1_1_1_1.xlsx" ]; then
    echo "[Step 0] 生成物理子集 ..."
    python 0514/scripts/ratio_ablation/prepare_subsampled_dataset.py
else
    echo "[Step 0] 子集已存在，跳过"
fi

# Step 1: 6 个新训练 (S2 = baseline 不训)
run_exp() {
    local LABEL=$1; local TRAIN=$2; local SAMPLER=$3
    echo "==================================================================="
    echo "[$(date '+%F %T')] 训练 $LABEL  sampler=$SAMPLER  train=$TRAIN"
    echo "==================================================================="
    python 0514/scripts/ratio_ablation/train_with_ratio.py \
        --exp_label "$LABEL" \
        --train_xlsx "$TRAIN" \
        --val_xlsx "$BASE_VAL" \
        --sampler "$SAMPLER" \
        --num_epochs "$NUM_EPOCHS"
}

# A 组 Sampler 消融 (固定数据集 = baseline train_split)
run_exp "S1_1_1_1"   "$BASE_TRAIN" "1_1_1"
run_exp "S3_1_2_6"   "$BASE_TRAIN" "1_2_6"
run_exp "S4_natural" "$BASE_TRAIN" "natural"

# B 组 Dataset 消融 (固定 sampler = natural)
run_exp "D1_1_1_1"   "$SUB_DIR/train_D1_1_1_1.xlsx"   "natural"
run_exp "D2_1_1_2"   "$SUB_DIR/train_D2_1_1_2.xlsx"   "natural"
run_exp "D3_1_1.9_3" "$SUB_DIR/train_D3_1_1.9_3.xlsx" "natural"

# Step 2: 汇总
echo "==================================================================="
echo "[$(date '+%F %T')] 汇总结果 ..."
echo "==================================================================="
python 0514/scripts/ratio_ablation/compare_all.py

echo ""
echo "✅ 全部完成。查看产物:"
echo "  0514/logs/ratio_ablation/_summary/"
