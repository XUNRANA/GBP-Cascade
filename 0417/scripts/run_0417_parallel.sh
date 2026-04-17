#!/bin/bash
set -e

cd /data1/ouyangxinglong/GBP-Cascade
SCRIPTS=0417/scripts
LOG=0417/logs

mkdir -p $LOG

echo "=== Phase B: 4-GPU 并发训练 ==="
echo "启动时间: $(date)"

# GPU0: 19-1 pAUC loss
CUDA_VISIBLE_DEVICES=0 python $SCRIPTS/20260417_task2_SwinV2Tiny_segcls_19_1.py \
    > $LOG/exp19_1_console.log 2>&1 &
PID0=$!
echo "  GPU0 → 19-1 (pAUC loss)  PID=$PID0"

# GPU1: 19-2 SAM+SWA
CUDA_VISIBLE_DEVICES=1 python $SCRIPTS/20260417_task2_SwinV2Tiny_segcls_19_2.py \
    > $LOG/exp19_2_console.log 2>&1 &
PID1=$!
echo "  GPU1 → 19-2 (SAM+SWA)    PID=$PID1"

# GPU2: 19-3 Mixup+TTA+MCDropout
CUDA_VISIBLE_DEVICES=2 python $SCRIPTS/20260417_task2_SwinV2Tiny_segcls_19_3.py \
    > $LOG/exp19_3_console.log 2>&1 &
PID2=$!
echo "  GPU2 → 19-3 (Mixup+TTA)  PID=$PID2"

# GPU3: 19-control seed1337
CUDA_VISIBLE_DEVICES=3 python $SCRIPTS/20260417_task2_SwinV2Tiny_segcls_19_control.py \
    --seed 1337 \
    > $LOG/exp19_ctrl_1337_console.log 2>&1 &
PID3=$!
echo "  GPU3 → control seed1337  PID=$PID3"

echo ""
echo "全部已启动，等待完成..."
echo "实时日志: tail -f $LOG/exp19_1_console.log"

# 等待所有进程，捕获退出码
FAILED=0
wait $PID0 || { echo "❌ 19-1 (GPU0) 失败，退出码=$?"; FAILED=1; }
echo "✓ 19-1 (GPU0) 完成  $(date)"
wait $PID1 || { echo "❌ 19-2 (GPU1) 失败，退出码=$?"; FAILED=1; }
echo "✓ 19-2 (GPU1) 完成  $(date)"
wait $PID2 || { echo "❌ 19-3 (GPU2) 失败，退出码=$?"; FAILED=1; }
echo "✓ 19-3 (GPU2) 完成  $(date)"
wait $PID3 || { echo "❌ ctrl-1337 (GPU3) 失败，退出码=$?"; FAILED=1; }
echo "✓ ctrl-1337 (GPU3) 完成  $(date)"

if [ $FAILED -eq 0 ]; then
    echo ""
    echo "=== Phase B 全部完成 $(date) ==="
    echo "下一步: bash 0417/scripts/run_0417_phase_c.sh"
else
    echo ""
    echo "=== Phase B 有失败任务，请检查日志 ==="
    exit 1
fi
