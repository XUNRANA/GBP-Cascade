#!/bin/bash
# 0322 改进实验 — 修复 Mixup 类权重 Bug + Balanced Sampler + Focal Loss + 阈值优化
source activate gbp
export CUDA_VISIBLE_DEVICES=1
cd /data1/ouyangxinglong/GBP-Cascade

echo "===== Exp #7 (重跑): SwinV2 Full4ch StrongAug+Mixup ★★★ ====="
python 20260323_task2_SwinV2Tiny_full4ch_strongaug_7.py

echo "===== Exp #9: SwinV2 Balanced + Fixed Mixup ★★★★ ====="
python 20260323_task2_SwinV2Tiny_balanced_mixup_9.py

echo "===== Exp #10: SwinV2 Focal + Threshold Tuning ★★★★★ ====="
python 20260323_task2_SwinV2Tiny_focal_threshold_10.py

echo "===== Exp #11: ConvNeXt Focal + Balanced ====="
python 20260323_task2_ConvNeXtTiny_balanced_focal_11.py

echo "===== Batch3 Done ====="
