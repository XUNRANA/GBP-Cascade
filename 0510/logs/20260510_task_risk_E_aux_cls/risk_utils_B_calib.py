"""实验 B 工具库 —— 非对称 ordinal target + 早停 + 温度校准。

设计目标
  baseline 在 epoch 6 就达到 best,后面 50+ epoch 全是过拟合;且校准曲线
  bimodal (中间段几乎没样本)。本实验用 3 个手段同时缓解:
    1. ordinal target 改为 (1.0, 0.35, 0.0) —— 把 benign 拉低,迫使模型学
       benign↔no_tumor 的细粒度差异
    2. 早停 (patience=5) —— val AUC 5 个 eval 周期不升即停
    3. 训练后做 temperature scaling —— 在 val 上用 LBFGS 拟合一个标量 T,
       让推理 score = sigmoid(logit / T),修复 bimodal

只新增工具,不新增模型类 —— 模型直接复用 baseline 的 SwinV2SegGuidedRiskTrimodal。
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn as nn


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)


# ═══════════════════════════════════════════════════════════════════════════
#  EarlyStopper
# ═══════════════════════════════════════════════════════════════════════════


class EarlyStopper:
    """val 指标 patience 个 eval 周期不升则触发 stop。

    用法:
        stopper = EarlyStopper(patience=5, mode="max")
        ...
        if stopper.step(val_auc):
            break  # 早停
    """

    def __init__(self, patience: int = 5, mode: str = "max",
                 min_delta: float = 1e-4):
        assert mode in ("max", "min")
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best = -np.inf if mode == "max" else np.inf
        self.bad_count = 0

    def _is_improvement(self, value: float) -> bool:
        if self.mode == "max":
            return value > self.best + self.min_delta
        return value < self.best - self.min_delta

    def step(self, value: float) -> bool:
        if self._is_improvement(value):
            self.best = value
            self.bad_count = 0
            return False
        self.bad_count += 1
        return self.bad_count >= self.patience


# ═══════════════════════════════════════════════════════════════════════════
#  TemperatureScaler
# ═══════════════════════════════════════════════════════════════════════════


class TemperatureScaler(nn.Module):
    """单标量温度缩放: score' = sigmoid(logit / T)。

    在 val (logit, binary_label=is_malignant) 上用 LBFGS 拟合 T,
    最小化 BCE-with-logits(logit / T, binary_label)。
    """

    def __init__(self, init_T: float = 1.0):
        super().__init__()
        # 用 log_T 参数化以保证 T > 0
        self.log_T = nn.Parameter(torch.log(torch.tensor(float(init_T))))

    @property
    def T(self) -> float:
        return float(torch.exp(self.log_T).item())

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / torch.exp(self.log_T)

    def fit(self, val_logits: np.ndarray, val_binary_labels: np.ndarray,
            max_iter: int = 200, lr: float = 0.05) -> float:
        """拟合 T,返回拟合后的 T 值。

        - val_logits: shape (N,),raw logits (sigmoid 之前)
        - val_binary_labels: shape (N,),0/1 (1 = malignant)
        """
        device = self.log_T.device
        logits = torch.tensor(val_logits, dtype=torch.float32, device=device)
        labels = torch.tensor(val_binary_labels, dtype=torch.float32, device=device)
        bce = nn.BCEWithLogitsLoss()

        optim = torch.optim.LBFGS([self.log_T], lr=lr, max_iter=max_iter)

        def _closure():
            optim.zero_grad()
            scaled = self.forward(logits)
            loss = bce(scaled, labels)
            loss.backward()
            return loss

        optim.step(_closure)
        return self.T


def apply_temperature(logits: np.ndarray, T: float) -> np.ndarray:
    """numpy 版温度校准: score = sigmoid(logit / T)。"""
    scaled = logits / max(T, 1e-6)
    # sigmoid (numpy)
    return 1.0 / (1.0 + np.exp(-scaled))


# ═══════════════════════════════════════════════════════════════════════════
#  Brier 评估辅助 (用于 log 校准前后对比)
# ═══════════════════════════════════════════════════════════════════════════


def quick_brier(scores: np.ndarray, labels: np.ndarray) -> float:
    """二值 Brier: target = is_malignant (label==0)。"""
    is_mal = (labels == 0).astype(np.float64)
    return float(np.mean((scores - is_mal) ** 2))
