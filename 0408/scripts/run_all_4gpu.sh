#!/bin/bash
# 四卡并行运行 Exp#15-18, nohup防断连
# 用法: bash run_all_4gpu.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$SCRIPT_DIR/../logs"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "  0408 四个实验并行启动 (4x A100)"
echo "=========================================="
echo "  GPU 0 → Exp#15 (10D扩展临床)"
echo "  GPU 1 → Exp#16 (BERT[CLS]后期融合)"
echo "  GPU 2 → Exp#17 (BERT交叉注意力)"
echo "  GPU 3 → Exp#18 (全模态门控融合)"
echo "=========================================="

# Exp#15 → GPU 0
CUDA_VISIBLE_DEVICES=0 nohup python "$SCRIPT_DIR/20260408_task2_SwinV2Tiny_segcls_15.py" \
    > "$LOG_DIR/exp15_console.log" 2>&1 &
PID15=$!
echo "[$(date '+%H:%M:%S')] Exp#15 started on GPU 0, PID=$PID15"

# Exp#16 → GPU 1
CUDA_VISIBLE_DEVICES=1 nohup python "$SCRIPT_DIR/20260408_task2_SwinV2Tiny_segcls_16.py" \
    > "$LOG_DIR/exp16_console.log" 2>&1 &
PID16=$!
echo "[$(date '+%H:%M:%S')] Exp#16 started on GPU 1, PID=$PID16"

# Exp#17 → GPU 2
CUDA_VISIBLE_DEVICES=2 nohup python "$SCRIPT_DIR/20260408_task2_SwinV2Tiny_segcls_17.py" \
    > "$LOG_DIR/exp17_console.log" 2>&1 &
PID17=$!
echo "[$(date '+%H:%M:%S')] Exp#17 started on GPU 2, PID=$PID17"

# Exp#18 → GPU 3
CUDA_VISIBLE_DEVICES=3 nohup python "$SCRIPT_DIR/20260408_task2_SwinV2Tiny_segcls_18.py" \
    > "$LOG_DIR/exp18_console.log" 2>&1 &
PID18=$!
echo "[$(date '+%H:%M:%S')] Exp#18 started on GPU 3, PID=$PID18"

echo ""
echo "=========================================="
echo "  全部启动完成! PID: $PID15 $PID16 $PID17 $PID18"
echo "=========================================="
echo ""
echo "监控命令:"
echo "  查看进度:  tail -f $LOG_DIR/exp15_console.log"
echo "  查看GPU:   nvidia-smi"
echo "  查看进程:  ps aux | grep segcls_1[5-8]"
echo "  等待完成:  wait $PID15 $PID16 $PID17 $PID18"
