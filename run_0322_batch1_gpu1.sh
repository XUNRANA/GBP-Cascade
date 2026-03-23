#!/bin/bash
# 0322 全部实验 — 使用 GPU1, conda env: gbp
source activate gbp
export CUDA_VISIBLE_DEVICES=1
cd /data1/ouyangxinglong/GBP-Cascade

echo "===== Exp #1: ConvNeXt-Tiny ROI (CNN Baseline) ====="
python 20260323_task2_ConvNeXtTiny_roi_1.py

echo "===== Exp #2: ResNet34 ROI (CNN Reference) ====="
python 20260323_task2_ResNet34_roi_2.py

echo "===== Exp #3: SwinV2-Tiny ROI+4ch ★ ====="
python 20260323_task2_SwinV2Tiny_roi4ch_3.py

echo "===== Exp #4: MaxViT-Tiny ROI+4ch ====="
python 20260323_task2_MaxViTTiny_roi4ch_4.py

echo "===== Exp #5: DeiT3-Small ROI+4ch ====="
python 20260323_task2_DeiT3Small_roi4ch_5.py

echo "===== Exp #6: SwinV2-Tiny Dual Branch ★★ ====="
python 20260323_task2_SwinV2Tiny_dual_6.py

echo "===== All Done ====="
