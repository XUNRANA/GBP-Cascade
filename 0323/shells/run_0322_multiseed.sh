#!/bin/bash
# 方案三 Step1: 5 种子训练
source activate gbp
export CUDA_VISIBLE_DEVICES=1
cd /data1/ouyangxinglong/GBP-Cascade

for i in 0 1 2 3 4; do
    echo "===== Seed $i / 5 ====="
    python 20260323_task2_multiseed_train.py $i
done

echo "===== All seeds done ====="
