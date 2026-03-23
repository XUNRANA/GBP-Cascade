#!/usr/bin/env bash
set -euo pipefail

cd /data1/ouyangxinglong/GBP-Cascade/0319
source /home/ubuntu/anaconda3/etc/profile.d/conda.sh
conda activate gbp

export CUDA_VISIBLE_DEVICES=1

python /data1/ouyangxinglong/GBP-Cascade/0319/20260319_task2_ConvNeXtTiny_adamw_warmupcosine_classweight_img320_seed84_7.py
python /data1/ouyangxinglong/GBP-Cascade/0319/20260319_task2_ConvNeXtTiny_adamw_warmupcosine_weightedsampler_img320_4.py
