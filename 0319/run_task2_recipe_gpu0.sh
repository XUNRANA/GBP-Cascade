#!/usr/bin/env bash
set -euo pipefail

cd /data1/ouyangxinglong/GBP-Cascade/0319
source /home/ubuntu/anaconda3/etc/profile.d/conda.sh
conda activate gbp

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

python 20260319_task2_EfficientnetB0_weakaug_classweight_warmupcosine_2.py
python 20260319_task2_ConvNeXtTiny_adamw_warmupcosine_classweight_2.py
