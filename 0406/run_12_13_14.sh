#!/bin/bash
# 在 GPU 3 上按顺序运行 Exp#12 → Exp#13 → Exp#14
PROJ="/data1/ouyangxinglong/GBP-Cascade"
S="$PROJ/0406/scripts"
export CUDA_VISIBLE_DEVICES=3

echo "===== Exp#12 开始: $(date) ====="
python "$S/20260406_task2_SwinV2Tiny_segcls_12.py"
echo "===== Exp#12 结束: $(date) ====="

echo "===== Exp#13 开始: $(date) ====="
python "$S/20260406_task2_SwinV2Tiny_segcls_13.py"
echo "===== Exp#13 结束: $(date) ====="

echo "===== Exp#14 开始: $(date) ====="
python "$S/20260406_task2_SwinV2Tiny_segcls_14.py"
echo "===== Exp#14 结束: $(date) ====="

echo "全部完成: $(date)"
