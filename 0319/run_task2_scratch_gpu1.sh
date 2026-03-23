#!/usr/bin/env bash
set -euo pipefail

cd /data1/ouyangxinglong/GBP-Cascade/0319
source /home/ubuntu/anaconda3/etc/profile.d/conda.sh
conda activate gbp

export CUDA_VISIBLE_DEVICES=1

python /data1/ouyangxinglong/GBP-Cascade/0319/20260319_task2_Resnet34_scratch_weakaug_classweight_6.py
