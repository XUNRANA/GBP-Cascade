#!/usr/bin/env bash
set -euo pipefail

cd /data1/ouyangxinglong/GBP-Cascade/0319
source /home/ubuntu/anaconda3/etc/profile.d/conda.sh
conda activate gbp

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=1

python 20260319_task2_SwinTiny_adamw_warmupcosine_classweight_2.py
python 20260319_task2_ViTB16_adamw_warmupcosine_classweight_2.py
