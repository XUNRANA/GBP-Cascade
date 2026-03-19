#!/usr/bin/env bash
set -euo pipefail

cd /data1/ouyangxinglong/GBP-Cascade
source /home/ubuntu/anaconda3/etc/profile.d/conda.sh
conda activate gbp

export CUDA_VISIBLE_DEVICES=1

python /data1/ouyangxinglong/GBP-Cascade/20260319_task2_SwinTiny_adamw_warmupcosine_focalloss_3.py
