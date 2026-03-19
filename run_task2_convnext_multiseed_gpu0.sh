#!/usr/bin/env bash
set -euo pipefail

cd /data1/ouyangxinglong/GBP-Cascade
source /home/ubuntu/anaconda3/etc/profile.d/conda.sh
conda activate gbp

export CUDA_VISIBLE_DEVICES=0

python /data1/ouyangxinglong/GBP-Cascade/20260319_task2_ConvNeXtTiny_adamw_warmupcosine_classweight_img320_seed7_5.py
python /data1/ouyangxinglong/GBP-Cascade/20260319_task2_ConvNeXtTiny_adamw_warmupcosine_classweight_img320_seed21_6.py
