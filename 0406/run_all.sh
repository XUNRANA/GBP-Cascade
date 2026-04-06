#!/bin/bash
# 0406 四个实验启动脚本
# Exp#11: 形态学特征扩展 (GPU 0)
# Exp#12: Val Split + Early Stopping + EMA (GPU 1)
# Exp#13: Feature-Level Mixup (GPU 2)
# Exp#14: 5-Fold CV + TTA (GPU 0, 建议 #11-13 完成后运行)

PROJ="/data1/ouyangxinglong/GBP-Cascade"
SCRIPTS="$PROJ/0406/scripts"

echo "============================================"
echo "0406 实验启动"
echo "============================================"

# --- 并行运行 Exp#11, #12, #13 ---

echo "[Exp#11] 形态学特征扩展 → GPU 0"
CUDA_VISIBLE_DEVICES=0 nohup python "$SCRIPTS/20260406_task2_SwinV2Tiny_segcls_11.py" \
    > "$PROJ/0406/logs/exp11_stdout.log" 2>&1 &
PID11=$!
echo "  PID=$PID11"

echo "[Exp#12] Val Split + Early Stopping + EMA → GPU 1"
CUDA_VISIBLE_DEVICES=1 nohup python "$SCRIPTS/20260406_task2_SwinV2Tiny_segcls_12.py" \
    > "$PROJ/0406/logs/exp12_stdout.log" 2>&1 &
PID12=$!
echo "  PID=$PID12"

echo "[Exp#13] Feature-Level Mixup → GPU 2"
CUDA_VISIBLE_DEVICES=2 nohup python "$SCRIPTS/20260406_task2_SwinV2Tiny_segcls_13.py" \
    > "$PROJ/0406/logs/exp13_stdout.log" 2>&1 &
PID13=$!
echo "  PID=$PID13"

echo ""
echo "并行运行中: Exp#11(PID=$PID11), Exp#12(PID=$PID12), Exp#13(PID=$PID13)"
echo "查看日志: tail -f $PROJ/0406/logs/exp1[1-3]_stdout.log"
echo ""
echo "Exp#14 (5-Fold CV + TTA) 建议在 #11-13 完成后单独运行:"
echo "  CUDA_VISIBLE_DEVICES=0 python $SCRIPTS/20260406_task2_SwinV2Tiny_segcls_14.py"
