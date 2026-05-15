#!/bin/bash
# run_parallel.sh — 在 4 GPU 上并行跑 6 个比例消融实验
#
# Wave 1: GPU 0/1/2/3 同时跑 S1, S3, S4, D3 (重量级)
# Wave 2: GPU 0/1 跑 D1, D2 (轻量)

set -e
cd /data1/ouyangxinglong/GBP-Cascade

NUM_EPOCHS=${1:-20}
BASE_VAL="/data1/ouyangxinglong/GBP-Cascade/0514/logs/20260514_task_risk_SwinV2Tiny_ordinal_trimodal_1/val_split.xlsx"
BASE_TRAIN="/data1/ouyangxinglong/GBP-Cascade/0514/logs/20260514_task_risk_SwinV2Tiny_ordinal_trimodal_1/train_split.xlsx"
SUB_DIR="/data1/ouyangxinglong/GBP-Cascade/0514dataset_flat/ratio_ablation_splits"
OUT_DIR="/data1/ouyangxinglong/GBP-Cascade/0514/logs/ratio_ablation"
mkdir -p "$OUT_DIR"

# 子集若不存在则生成
if [ ! -f "$SUB_DIR/train_D1_1_1_1.xlsx" ]; then
    python 0514/scripts/ratio_ablation/prepare_subsampled_dataset.py
fi

run_bg() {
    local GPU=$1; local LABEL=$2; local TRAIN=$3; local SAMPLER=$4
    local OUT_LOG="$OUT_DIR/${LABEL}_console.log"
    echo "[$(date '+%H:%M:%S')] GPU $GPU → $LABEL"
    CUDA_VISIBLE_DEVICES=$GPU python 0514/scripts/ratio_ablation/train_with_ratio.py \
        --exp_label "$LABEL" \
        --train_xlsx "$TRAIN" \
        --val_xlsx "$BASE_VAL" \
        --sampler "$SAMPLER" \
        --num_epochs "$NUM_EPOCHS" > "$OUT_LOG" 2>&1 &
    echo $!
}

# ── Wave 1: 4 个重量级实验 ──
echo "============ Wave 1: S1/S3/S4/D3 在 4 GPU 并行 ============"
PIDS=()
PIDS+=( $(run_bg 0 "S1_1_1_1"   "$BASE_TRAIN" "1_1_1") )
PIDS+=( $(run_bg 1 "S3_1_2_6"   "$BASE_TRAIN" "1_2_6") )
PIDS+=( $(run_bg 2 "S4_natural" "$BASE_TRAIN" "natural") )
PIDS+=( $(run_bg 3 "D3_1_1.9_3" "$SUB_DIR/train_D3_1_1.9_3.xlsx" "natural") )

echo "等待 Wave 1 完成 (PIDs: ${PIDS[@]}) ..."
for PID in "${PIDS[@]}"; do
    wait "$PID" || echo "PID $PID 失败"
done
echo "[$(date '+%H:%M:%S')] Wave 1 完成"

# ── Wave 2: 2 个小数据集 ──
echo "============ Wave 2: D1/D2 在 GPU 0/1 并行 ============"
PIDS=()
PIDS+=( $(run_bg 0 "D1_1_1_1" "$SUB_DIR/train_D1_1_1_1.xlsx" "natural") )
PIDS+=( $(run_bg 1 "D2_1_1_2" "$SUB_DIR/train_D2_1_1_2.xlsx" "natural") )

echo "等待 Wave 2 完成 (PIDs: ${PIDS[@]}) ..."
for PID in "${PIDS[@]}"; do
    wait "$PID" || echo "PID $PID 失败"
done
echo "[$(date '+%H:%M:%S')] Wave 2 完成"

# ── 汇总 ──
echo "============ 汇总结果 ============"
python 0514/scripts/ratio_ablation/compare_all.py

echo ""
echo "✅ 全部完成。查看产物:"
echo "  $OUT_DIR/_summary/"
