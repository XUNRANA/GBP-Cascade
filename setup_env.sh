#!/bin/bash
# ============================================================
# GBP-Cascade 虚拟环境一键搭建脚本
# 环境要求：Anaconda + CUDA 12.x (已验证 CUDA 12.8)
# 创建 Python 3.11 conda 环境 + PyTorch 2.5 (cu124)
# ============================================================

set -e  # 任意步骤出错立即退出

ENV_NAME="gbp"
PYTHON_VER="3.11"

echo "========================================"
echo " GBP-Cascade 环境安装脚本"
echo " 环境名: $ENV_NAME | Python: $PYTHON_VER"
echo "========================================"

# -------- 1. 创建 conda 环境 --------
echo ""
echo "[1/4] 创建 conda 环境: $ENV_NAME ..."
conda create -y -n $ENV_NAME python=$PYTHON_VER

# 激活环境（在脚本中需要 source）
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

echo "     当前 Python: $(python --version)"

# -------- 2. 安装 PyTorch (CUDA 12.4，兼容 CUDA 12.x 驱动) --------
echo ""
echo "[2/4] 安装 PyTorch 2.5 + CUDA 12.4 ..."
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

# -------- 3. 安装项目依赖 --------
echo ""
echo "[3/4] 安装项目依赖包 ..."
pip install \
    scikit-learn==1.5.2 \
    Pillow==11.0.0 \
    tqdm==4.67.1 \
    numpy==1.26.4 \
    matplotlib==3.9.3 \
    pandas==2.2.3

# -------- 4. 验证安装 --------
echo ""
echo "[4/4] 验证安装结果 ..."
python - <<'PYEOF'
import torch
import torchvision
from PIL import Image
import sklearn
import tqdm
import numpy

print(f"  PyTorch      : {torch.__version__}")
print(f"  torchvision  : {torchvision.__version__}")
print(f"  scikit-learn : {sklearn.__version__}")
print(f"  Pillow       : {Image.__version__}")
print(f"  NumPy        : {numpy.__version__}")
print(f"  CUDA 可用    : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        mem  = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"  GPU {i}        : {name} ({mem:.0f} GB)")

# 验证 ConvNeXt-Base 可以加载
from torchvision import models
m = models.convnext_base(weights=None)
print(f"  ConvNeXt-Base: OK (参数量 {sum(p.numel() for p in m.parameters())/1e6:.1f}M)")
PYEOF

echo ""
echo "========================================"
echo " 安装完成！"
echo ""
echo " 激活环境："
echo "   conda activate $ENV_NAME"
echo ""
echo " 运行训练："
echo "   python train_convnext.py"
echo "   python train_dinov2.py"
echo ""
echo " DINOv2 注意：首次运行 train_dinov2.py 会自动"
echo " 从 torch.hub 下载约 330MB 预训练权重"
echo "========================================"
