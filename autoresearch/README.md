# GBP-Cascade AutoResearch

> **AI Agent 自主实验框架 — 胆囊息肉超声图像分类**

借鉴 [Karpathy 的 autoresearch](https://github.com/karpathy/autoresearch) 理念，为胆囊息肉超声分类任务（Task 2: 良性肿瘤 vs 非肿瘤性息肉）搭建的**自主实验平台**。AI Agent 在你睡觉时自动修改代码、训练模型、评估指标、决定保留或丢弃，循环往复。你醒来时看到一份完整的实验日志和（如果运气好的话）更好的模型。

---

## 目录

- [核心理念](#核心理念)
- [项目结构](#项目结构)
- [工作原理](#工作原理)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [启动 Agent 自主实验](#启动-agent-自主实验)
- [train_task2.py 详解](#train_task2py-详解)
- [prepare_task2.py 详解](#prepare_task2py-详解)
- [数据集说明](#数据集说明)
- [评估指标](#评估指标)
- [实验循环详解](#实验循环详解)
- [结果分析](#结果分析)
- [前期实验经验](#前期实验经验)
- [Agent 搜索策略建议](#agent-搜索策略建议)
- [设计决策](#设计决策)
- [常见问题](#常见问题)
- [致谢](#致谢)

---

## 核心理念

传统的深度学习实验流程是：

```
人类想一个方案 → 手动改代码 → 跑实验 → 等结果 → 分析 → 再想一个方案 → ...
```

这个循环很慢——每天最多跑 5-10 个实验，且受限于人的精力和想象力。

**AutoResearch 的理念是把这个循环交给 AI Agent：**

```
Agent 提出方案 → 自动改代码 → 训练 → 评估 →
  ├─ 指标提升 → 保留修改 (git keep)
  └─ 指标下降 → 回退修改 (git reset)
  └─ 永远循环，直到人类手动停止
```

**核心优势：**

| | 人工实验 | AutoResearch |
|---|---|---|
| 每天实验数量 | 5-10 个 | **60-100 个** |
| 实验时间 | 仅工作时间 | **24/7 不间断** |
| 搜索空间 | 受限于经验 | **系统化穷举** |
| 实验记录 | 手动整理 | **自动 TSV + Git** |
| 可复现性 | 依赖笔记 | **完整 Git 历史** |

---

## 项目结构

```
autoresearch/
├── train_task2.py        # [Agent 修改] 模型 + 超参 + 训练循环 (280 行)
├── prepare_task2.py      # [只读固定] 数据集 + Transform + 评估函数 (553 行)
├── program_gbp.md        # [人类编写] Agent 的任务说明书 (144 行)
├── analysis_gbp.py       # [人类使用] 实验结果分析 + 可视化 (100 行)
├── .gitignore            # Git 忽略规则
├── README.md             # 本文档
│
├── results.tsv           # [自动生成] 实验记录表 (不提交到 Git)
├── run.log               # [自动生成] 最近一次训练日志 (不提交)
├── _best_model.pth       # [自动生成] 当前实验最优权重 (不提交)
└── progress.png          # [自动生成] 实验进展图 (不提交)
```

**关键设计：只有 `train_task2.py` 可以被 Agent 修改。** 其余文件要么固定（`prepare_task2.py`），要么由人类编写（`program_gbp.md`），要么自动生成。

---

## 工作原理

### 整体架构

```
┌──────────────────────────────────────────────────────┐
│                  program_gbp.md                       │
│              (Agent 的 "任务说明书")                    │
│      定义目标、约束、搜索方向、实验协议                    │
└──────────────────┬───────────────────────────────────┘
                   │ 指导
                   ▼
┌──────────────────────────────────────────────────────┐
│                 AI Agent (Claude Code 等)              │
│                                                       │
│  循环:                                                │
│  1. 读取 train_task2.py                               │
│  2. 提出修改方案 (改超参/换模型/调增强)                   │
│  3. 修改 train_task2.py                               │
│  4. git commit                                        │
│  5. python train_task2.py > run.log 2>&1              │
│  6. grep 提取 f1_at_threshold                         │
│  7. f1 提升? → keep : git reset                       │
│  8. 记录到 results.tsv                                │
│  9. 回到第 1 步                                       │
└──────────┬─────────────────────────────┬─────────────┘
           │ 调用                         │ 调用
           ▼                             ▼
┌─────────────────────┐    ┌─────────────────────────┐
│  train_task2.py     │    │  prepare_task2.py       │
│  (Agent 可修改)      │    │  (固定不可修改)          │
│                     │    │                         │
│  - 超参数           │───▶│  - 数据集类              │
│  - 模型选择          │    │  - Transform            │
│  - 训练循环          │    │  - 评估函数              │
│  - 优化器配置        │    │  - 工具函数              │
│  - 损失函数          │    │  - 固定常量              │
└─────────────────────┘    └─────────────────────────┘
           │                             │
           ▼                             ▼
┌──────────────────────────────────────────────────────┐
│                    0322dataset/                        │
│           (1229 训练 + 523 测试, 固定划分)               │
│       benign (良性肿瘤, 309) + no_tumor (非肿瘤, 920)   │
└──────────────────────────────────────────────────────┘
```

### 单次实验流程

```
python train_task2.py
  │
  ├─ 1. 初始化: 设定随机种子, 打印配置
  │
  ├─ 2. 构建模型: timm 预训练模型 → 4通道适配 → GPU
  │
  ├─ 3. 构建数据: 选择数据集类 + Transform → DataLoader
  │
  ├─ 4. 构建优化器: AdamW + 差异化学习率 (backbone 慢, head 快)
  │
  ├─ 5. 训练循环: (最多 NUM_EPOCHS 轮, 受 TIME_BUDGET 限制)
  │     │
  │     ├─ 每轮: 前向 → Mixup/CutMix → 计算 Loss → 反向 → 梯度裁剪 → 更新
  │     │
  │     └─ 每 EVAL_INTERVAL 轮: 评估 → 保存最优 checkpoint
  │
  ├─ 6. 加载最优 checkpoint
  │
  ├─ 7. 最终评估: evaluate_model() → 含阈值搜索
  │
  └─ 8. 输出指标 (grep 友好格式):
        ---
        f1_at_threshold:       0.624000    ← 主要指标
        best_threshold:        0.530
        f1_macro:              0.617300
        peak_vram_mb:          4500.2
        num_params_M:          28.3
```

---

## 环境要求

- **GPU**: NVIDIA GPU (已在 RTX 3090 / A100 / H100 上测试)
- **Python**: 3.10+
- **核心依赖**:
  - `torch` (含 CUDA)
  - `timm` (预训练模型库)
  - `scikit-learn` (评估指标)
  - `pandas`, `numpy`, `Pillow` (数据处理)
  - `matplotlib` (结果可视化)
  - `openpyxl` (读取 Excel 数据划分)

安装：

```bash
pip install torch torchvision timm scikit-learn pandas numpy Pillow matplotlib openpyxl
```

---

## 快速开始

### 1. 验证数据集

确认数据集存在于上层目录：

```bash
ls ../0322dataset/
# 应包含: benign/  no_tumor/  malignant/  task_2_train.xlsx  task_2_test.xlsx
```

### 2. 验证环境

```bash
cd /data1/ouyangxinglong/GBP-Cascade/autoresearch

# 检查 prepare_task2.py 能否正常导入
python -c "from prepare_task2 import *; print(f'Data: {DATA_ROOT}'); print('OK')"
```

### 3. 手动运行一次基线实验

```bash
# 运行完整训练 (~5-8 分钟)
python train_task2.py
```

你应该看到类似输出：

```
=== GBP-Cascade Task 2 AutoResearch ===
Model: swinv2_tiny_window8_256 | Input: full_4ch | Size: 256 | Aug: strong
...
Epoch 005/100 | Loss: 0.8234 | Acc: 0.6532 | LR: 0.6250 | 3.2s | F1@thresh: 0.5432 (NEW BEST*)
...
Epoch 100/100 | Loss: 0.4123 | Acc: 0.8921 | LR: 0.0100 | 3.1s | F1@thresh: 0.6173 (best: 0.6240@ep85)
---
f1_at_threshold:       0.624000
best_threshold:        0.530
...
```

### 4. 初始化 Git

```bash
git init
git add prepare_task2.py train_task2.py program_gbp.md analysis_gbp.py .gitignore README.md
git commit -m "initial autoresearch framework for GBP-Cascade Task 2"
```

---

## 启动 Agent 自主实验

### 方式一：使用 Claude Code (推荐)

```bash
# 1. 在 autoresearch 目录启动 Claude Code
cd /data1/ouyangxinglong/GBP-Cascade/autoresearch
claude

# 2. 提示 Agent 开始实验
# 输入:
Hi, have a look at program_gbp.md and let's kick off a new experiment! Let's do the setup first.
```

Agent 会自动：
- 阅读 `program_gbp.md` 了解实验协议
- 阅读 `train_task2.py` 了解当前配置
- 创建实验分支 `autoresearch/<tag>`
- 初始化 `results.tsv`
- 开始自主实验循环

**然后你就可以去睡觉了。** Agent 会持续运行，直到你手动停止。

### 方式二：使用其他 AI Agent

任何支持代码编辑和命令执行的 AI Agent 都可以。关键是：
1. 将 `program_gbp.md` 的内容作为 Agent 的上下文/系统提示
2. 确保 Agent 有权限编辑文件和执行 shell 命令
3. 禁用所有交互式确认（Agent 需要完全自主运行）

### 方式三：手动模拟 Agent 循环

如果你想手动跑几个实验试试：

```bash
# 创建分支
git checkout -b autoresearch/manual-test

# 创建结果表
echo -e "commit\tf1_at_threshold\tpeak_vram_mb\tstatus\tdescription" > results.tsv

# 跑基线
python train_task2.py > run.log 2>&1
grep "^f1_at_threshold:\|^peak_vram_mb:" run.log
# → 记录到 results.tsv

# 修改超参 (例如增大 dropout)
# vim train_task2.py  → 修改 DROP_RATE = 0.4
git add train_task2.py && git commit -m "increase dropout to 0.4"

# 跑实验
python train_task2.py > run.log 2>&1
grep "^f1_at_threshold:" run.log

# 如果结果变差 → 回退
git reset --hard HEAD~1

# 如果结果变好 → 保留, 继续下一个实验
```

---

## train_task2.py 详解

这是 Agent **唯一可以修改**的文件。它包含 5 个主要部分：

### 1. 超参数区域 (第 38-74 行)

所有可调参数集中在文件顶部，分为 6 组：

```python
# ===== Model =====
MODEL_NAME = "swinv2_tiny_window8_256"   # timm 模型名 (数百种可选)
DROP_RATE = 0.3                           # 分类头 dropout

# ===== Input =====
INPUT_MODE = "full_4ch"   # 输入模式:
                          #   "roi_3ch"  — ROI裁剪, 3通道RGB
                          #   "roi_4ch"  — ROI裁剪, 4通道RGB+mask
                          #   "full_4ch" — 全图, 4通道RGB+mask (推荐)
IMG_SIZE = 256            # 输入尺寸 (正方形)
AUG_MODE = "strong"       # 增强模式: "weak" | "strong"

# ===== Training =====
BATCH_SIZE = 8
NUM_EPOCHS = 100
EVAL_INTERVAL = 5         # 每 N 轮评估一次
WARMUP_EPOCHS = 8         # 学习率预热轮数
SEED = 42

# ===== Optimizer =====
BACKBONE_LR = 2e-5        # 骨干网络学习率 (慢)
HEAD_LR = 2e-4            # 分类头学习率 (快, 10x backbone)
WEIGHT_DECAY = 5e-2
MIN_LR_RATIO = 0.01       # 余弦衰减最低 LR 比例
GRAD_CLIP = 1.0            # 梯度裁剪阈值

# ===== Loss =====
LABEL_SMOOTHING = 0.1
USE_FOCAL_LOSS = False     # True → 使用 Focal Loss (替代 CE)
FOCAL_GAMMA = 2.0

# ===== Regularization =====
USE_MIXUP = True           # Mixup + CutMix 数据增强
MIXUP_ALPHA = 0.4          # Mixup Beta 分布参数
CUTMIX_ALPHA = 1.0         # CutMix Beta 分布参数
USE_BALANCED_SAMPLER = True # WeightedRandomSampler 类别均衡
```

### 2. 模型构建 (第 88-93 行)

```python
model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=2, drop_rate=DROP_RATE)
if INPUT_MODE in ("roi_4ch", "full_4ch"):
    adapt_model_to_4ch(model)   # 第一层 Conv2d: 3ch → 4ch
model = model.to(device)
```

Agent 可以：
- 换任何 `timm` 预训练模型
- 调整 `drop_rate`
- 添加自定义 head
- 冻结/解冻特定层

### 3. 数据加载 (第 99-119 行)

根据 `INPUT_MODE` 选择数据集类和 Transform：

| `INPUT_MODE` | 数据集类 | 输入通道 | 说明 |
|---|---|---|---|
| `roi_3ch` | `GBPDatasetROI` | 3 (RGB) | 胆囊 ROI 裁剪, 标准 Transform |
| `roi_4ch` | `GBPDatasetROI4ch` | 4 (RGB+mask) | ROI 裁剪 + 病灶 mask |
| `full_4ch` | `GBPDatasetFull4ch` | 4 (RGB+mask) | 全图 + 病灶 mask (推荐) |

| `AUG_MODE` | Transform 类 | 增强强度 |
|---|---|---|
| `weak` | `SyncTransform` | HFlip, VFlip, 小角度仿射, 轻微色彩 |
| `strong` | `StrongSyncTransform` | RandomResizedCrop, 大角度旋转, shear, 强色彩, 高斯模糊, RandomErasing, 高斯噪声 |

### 4. 训练循环 (第 136-187 行)

```
for epoch in range(1, NUM_EPOCHS + 1):
    ├─ 时间预算检查 (超过 TIME_BUDGET=600s 则停止)
    ├─ 余弦学习率衰减 (含线性 warmup)
    ├─ 训练一个 epoch:
    │   ├─ 可选 Mixup/CutMix (soft labels + class weight)
    │   ├─ AMP 混合精度前向
    │   ├─ 梯度裁剪 + 优化器步进
    │   └─ NaN 检测 (快速失败)
    └─ 每 EVAL_INTERVAL 轮: 评估 + 保存最优 checkpoint
```

**关键 bug fix**: Mixup 的 soft cross-entropy 必须包含 class weight (第 163-168 行)。这个 bug 曾导致 benign recall 从 42% 暴跌到 20%。

### 5. 输出格式 (第 200-218 行)

训练结束后打印 grep 友好的指标：

```
---
f1_at_threshold:       0.624000    ← 主要对比指标 (越高越好)
best_threshold:        0.530       ← 最优分类阈值
f1_macro:              0.617300    ← 默认阈值(0.5)的 F1
accuracy:              0.780000
f1_benign:             0.580000    ← 良性类 F1
f1_no_tumor:           0.654600    ← 非肿瘤类 F1
recall_benign:         0.418600    ← 良性类召回率
recall_no_tumor:       0.850000
precision_benign:      0.425200    ← 良性类精确率
precision_no_tumor:    0.880000
f1_benign_thresh:      0.600000    ← 最优阈值下的良性 F1
recall_benign_thresh:  0.500000    ← 最优阈值下的良性召回
best_epoch:            85          ← 最优模型所在 epoch
epochs_completed:      100
training_seconds:      312.5       ← 训练总耗时
peak_vram_mb:          4500.2      ← GPU 显存峰值
num_params_M:          28.3        ← 模型参数量 (百万)
```

---

## prepare_task2.py 详解

这是**固定不可修改**的基础设施文件，包含 6 个部分：

| Part | 内容 | 主要导出 |
|---|---|---|
| **Constants** | 数据路径, 类别名, 时间预算 | `DATA_ROOT`, `TRAIN_EXCEL`, `TEST_EXCEL`, `CLASS_NAMES`, `NUM_CLASSES`, `TIME_BUDGET`, `CHECKPOINT_PATH` |
| **Part 1** | JSON 标注解析 | `load_annotation()`, `get_gallbladder_rect()`, `generate_lesion_mask()`, `crop_roi()` |
| **Part 2** | 数据增强 Transform | `SyncTransform`, `StrongSyncTransform`, `build_roi_train_transform()`, `build_roi_test_transform()` |
| **Part 3** | 数据集类 | `GBPDatasetROI`, `GBPDatasetROI4ch`, `GBPDatasetFull4ch` |
| **Part 4** | 模型工具 | `adapt_model_to_4ch()`, `split_backbone_and_head()` |
| **Part 5** | 训练工具 | `set_seed()`, `build_class_weights()`, `build_weighted_sampler()`, `FocalLoss`, `cosine_warmup_factor()`, `mixup_cutmix_data()` |
| **Part 6** | 评估函数 (Ground Truth) | `evaluate_model()` |

### 为什么 `prepare_task2.py` 不可修改？

1. **评估公平性**: `evaluate_model()` 是所有实验的唯一评判标准。如果 Agent 能修改评估逻辑，就可能"作弊"
2. **数据一致性**: 数据集类确保所有实验使用完全相同的数据读取方式
3. **降低风险**: Agent 修改的范围越小，出错概率越低
4. **Git diff 可读性**: 只有一个文件变化，代码审查更方便

---

## 数据集说明

### 基本信息

| 属性 | 值 |
|---|---|
| 来源 | 胆囊超声图像 |
| 任务 | 二分类: 良性肿瘤(benign) vs 非肿瘤性息肉(no_tumor) |
| 训练集 | 1229 张 (benign=309, no_tumor=920) |
| 测试集 | 523 张 (benign=129, no_tumor=394) |
| 类别比 | 1:3 (严重不平衡) |
| 图像尺寸 | ~320x320 像素 |
| 标注格式 | LabelMe JSON |

### 标注内容

每张图像对应一个 JSON 标注文件，包含：

| 标注类型 | 形状 | 说明 |
|---|---|---|
| `gallbladder` | 矩形 (rectangle) | 胆囊区域包围框 |
| `gallbladder polyp` | 多边形 (polygon) | 息肉/病灶轮廓 |
| `pred` | 多边形 | 另一处病灶区域 |
| `gallbladder adenoma` | 多边形 | 腺瘤病灶 |
| `gallbladder tubular adenoma` | 多边形 | 管状腺瘤 |

### 数据使用方式

框架提供 3 种数据使用策略：

```
策略 A: roi_3ch
  原图 → gallbladder 矩形裁剪 → Resize → [3, H, W]
  (仅保留胆囊区域)

策略 B: roi_4ch
  原图 → gallbladder 矩形裁剪 → Resize → RGB [3, H, W]
  标注 → 病灶多边形 → 二值 mask → 裁剪 → Resize → Mask [1, H, W]
  拼接 → [4, H, W]

策略 C: full_4ch (推荐)
  原图 → Resize → RGB [3, H, W]
  标注 → 病灶多边形 → 二值 mask → Resize → Mask [1, H, W]
  拼接 → [4, H, W]
  (保留全局上下文 + 病灶位置信息)
```

### 为什么这个任务很难？

1. **样本极少**: 良性类仅 309 张训练图像
2. **视觉极相似**: 两类病灶面积占比仅相差 ~0.7% (benign 2.09% vs no_tumor 1.39%)
3. **超声图像质量**: 超声图像固有的噪声和伪影
4. **标注局限**: JSON 标注只提供位置/形状信息，无法反映组织学特征

---

## 评估指标

### 主要指标: `f1_at_threshold`

**F1(macro) with optimal threshold search.**

评估流程：
1. 模型对测试集所有样本输出 `P(benign)` 概率
2. 在 `[0.15, 0.75]` 范围内搜索最优分类阈值 (步长 0.005)
3. 对每个阈值: `P(benign) >= threshold → predict benign, else → predict no_tumor`
4. 计算 `F1(macro)` = `(F1_benign + F1_no_tumor) / 2`
5. 返回最高 F1 对应的阈值和 F1 值

### 为什么用阈值搜索？

默认阈值 0.5 对类别不平衡任务不是最优的。阈值搜索通常能提升 1-2% F1。例如：
- 默认阈值 0.5 → F1 = 0.617
- 最优阈值 0.53 → F1 = 0.624 (+0.7%)

### 全部输出指标

| 指标 | 含义 | 对应输出字段 |
|---|---|---|
| **F1 at threshold** | 最优阈值下的 F1(macro) | `f1_at_threshold` |
| Best threshold | 最优分类阈值 | `best_threshold` |
| F1 macro | 默认阈值(0.5)的 F1(macro) | `f1_macro` |
| Accuracy | 准确率 | `accuracy` |
| F1 benign | 良性类 F1 (默认阈值) | `f1_benign` |
| F1 no_tumor | 非肿瘤类 F1 (默认阈值) | `f1_no_tumor` |
| Recall benign | 良性类召回率 | `recall_benign` |
| Precision benign | 良性类精确率 | `precision_benign` |
| F1 benign @thresh | 最优阈值下良性类 F1 | `f1_benign_thresh` |
| Recall benign @thresh | 最优阈值下良性类召回 | `recall_benign_thresh` |
| Best epoch | 最优 checkpoint 所在 epoch | `best_epoch` |
| Training time | 训练总耗时 (秒) | `training_seconds` |
| Peak VRAM | GPU 显存峰值 (MB) | `peak_vram_mb` |
| Model size | 模型参数量 (百万) | `num_params_M` |

---

## 实验循环详解

### 分支策略

每次实验会话使用独立 Git 分支：

```
main ──────────────────────────────────────
  │
  └── autoresearch/mar24 ──┬──┬──┬──┬──→
                            │  │  │  │
                            │  │  │  └─ exp4: keep (F1=0.632)
                            │  │  └──── exp3: discard (reset)
                            │  └─────── exp2: keep (F1=0.628)
                            └────────── exp1: keep (baseline, F1=0.624)
```

- **keep**: commit 保留在分支上，后续实验基于此继续
- **discard**: `git reset --hard HEAD~1` 回退，commit 被丢弃
- **crash**: 记录错误，尝试修复或跳过

### 决策逻辑

```python
if new_f1_at_threshold > previous_best_f1_at_threshold:
    status = "keep"       # 保留 commit, 更新 best
else:
    status = "discard"    # git reset --hard HEAD~1
```

### results.tsv 格式

```
commit	f1_at_threshold	peak_vram_mb	status	description
a1b2c3d	0.624000	4500.2	keep	baseline (swinv2 + full4ch + strong aug)
b2c3d4e	0.630100	4600.0	keep	dropout 0.4 + backbone_lr 1e-5
c3d4e5f	0.618000	3200.0	discard	switched to resnet50 (worse)
d4e5f6g	0.000000	0.0	crash	convnext_large (OOM)
e5f6g7h	0.632500	4800.0	keep	efficientnet_b3 + focal_gamma=3
```

注意：`results.tsv` **不提交到 Git**，仅在本地追踪。所有实验变化通过 Git commit 历史可以回溯。

---

## 结果分析

实验结束后，使用分析脚本查看结果：

```bash
python analysis_gbp.py
```

输出示例：

```
==================================================
GBP-Cascade Task 2 AutoResearch Results
==================================================
Total experiments:  47
  Kept:             12 (26%)
  Discarded:        31 (66%)
  Crashed:          4 (9%)

Best f1_at_threshold: 0.648200
  Commit:           e5f6g7h
  VRAM:             4800.0 MB
  Description:      efficientnet_b3 + focal_gamma=3 + mixup_alpha=0.2

Progression (kept experiments):
  a1b2c3d  F1=0.624000  VRAM=4500MB  baseline <-- NEW BEST
  b2c3d4e  F1=0.630100  VRAM=4600MB  dropout 0.4 <-- NEW BEST
  ...
```

同时生成 `progress.png` 可视化图表：
- 上图: F1 随实验进展的变化 (绿色=keep, 红色=discard, 灰色=crash)
- 下图: 各实验 GPU 显存使用

---

## 前期实验经验

在搭建此框架之前，已手动完成 11 个实验。以下是关键发现：

### 已验证的结论

| 结论 | 证据 |
|---|---|
| **4通道 > 3通道** | ROI 4ch (F1=0.617) > ROI 3ch (F1=0.589-0.603) |
| **强增强 > 弱增强** | Strong aug (F1=0.617) > Weak aug (F1=0.583-0.617) |
| **Swin-V2 最稳定** | Swin-V2 (F1=0.617-0.624) vs ConvNeXt (0.589-0.613) vs MaxViT (0.583) |
| **BalancedSampler 有效** | 提升 benign recall 约 5-8% |
| **阈值搜索必要** | 提升 F1 约 1-2% |
| **Mixup class weight 关键** | 修复后 benign recall: 20% → 42% |

### 当前最优配置 (Exp #9)

```python
MODEL_NAME = "swinv2_tiny_window8_256"
INPUT_MODE = "full_4ch"
AUG_MODE = "strong"
USE_MIXUP = True          # with class weight fix
USE_BALANCED_SAMPLER = True
BACKBONE_LR = 2e-5
HEAD_LR = 2e-4
DROP_RATE = 0.3
NUM_EPOCHS = 100
# → f1_at_threshold ≈ 0.624
```

### 集成学习结果 (参考)

最佳双模型集成 (Exp #3 + Exp #9): F1 = 0.642。说明模型多样性有一定帮助，但单模型能力仍是瓶颈。

---

## Agent 搜索策略建议

### 优先级排序

以下按预期收益从高到低排列：

#### Tier 1: 高收益方向

1. **数据增强参数精调** — 组合空间巨大，人工难以穷举
   - `MIXUP_ALPHA`: 尝试 0.1, 0.2, 0.3, 0.5, 0.8
   - `CUTMIX_ALPHA`: 尝试 0.5, 1.0, 2.0
   - StrongSyncTransform 中的 crop scale, rotation angle, noise level
   - 添加新增强: GridDistortion, ElasticTransform

2. **类别平衡策略** — 核心瓶颈之一
   - `FOCAL_GAMMA`: 尝试 1.0, 1.5, 2.0, 3.0, 5.0
   - Class weight 缩放因子 (e.g., `class_weights ** 0.5`)
   - 过采样 benign + 欠采样 no_tumor 组合

3. **学习率和调度** — 对收敛质量影响大
   - `BACKBONE_LR`: 尝试 1e-5 到 5e-5
   - `HEAD_LR` / `BACKBONE_LR` 比值: 尝试 5x, 10x, 20x
   - `WARMUP_EPOCHS`: 尝试 3, 5, 8, 15
   - `MIN_LR_RATIO`: 尝试 0.001, 0.01, 0.05

#### Tier 2: 中等收益方向

4. **模型选择** — 不同架构可能有惊喜
   - `efficientnet_b2`, `efficientnet_b3` (CNN, 更高效)
   - `convnext_small.fb_in22k` (IN-22K 预训练)
   - `eva02_small_patch14_336.mim_in22k_ft_in1k` (MIM 预训练)
   - `caformer_s18.sail_in22k_ft_in1k` (MetaFormer)

5. **正则化组合** — 抗过拟合关键
   - `DROP_RATE`: 尝试 0.1, 0.2, 0.3, 0.4, 0.5
   - `WEIGHT_DECAY`: 尝试 0.01, 0.03, 0.05, 0.1
   - `LABEL_SMOOTHING`: 尝试 0.0, 0.05, 0.1, 0.2
   - Stochastic depth (通过 timm 的 `drop_path_rate` 参数)

#### Tier 3: 探索性方向

6. **输入表示** — 改变模型看到的信息
   - `IMG_SIZE`: 尝试 224, 288, 320, 384
   - ROI padding ratio: 尝试 0.0, 0.02, 0.05, 0.1
   - Mask 通道权重 (乘以 2.0 或 0.5)

7. **训练技巧** — 高级优化
   - EMA (Exponential Moving Average)
   - 梯度累积 (模拟更大 batch size)
   - 余弦重启 (Cosine Annealing with Warm Restarts)
   - 知识蒸馏 (从大模型到小模型)

---

## 设计决策

### 为什么只有一个文件可修改？

参考 Karpathy 的 autoresearch 设计：

> "The agent only touches `train.py`. This keeps the scope manageable and diffs reviewable."

对于 AI Agent 来说，修改范围越小，出错概率越低。Agent 的所有创造力都集中在一个文件里，每次 commit 的 diff 清晰可读。

### 为什么用固定时间预算？

每个实验最多训练 10 分钟 (`TIME_BUDGET = 600`)。这保证：
- 实验速度一致，可以预估总实验数量
- Agent 不会陷入"多训几轮可能会更好"的陷阱
- 不同模型/配置的结果直接可比

### 为什么用 epoch-based 而不是 time-based 训练？

与 autoresearch (语言模型, time-based) 不同，图像分类任务：
- 数据集小 (1229 张), 每个 epoch 很快 (~3s)
- 收敛取决于 epoch 数，而非训练时间
- 需要周期性评估 + 保存最优 checkpoint (防止过拟合)

因此采用 epoch-based 训练 + time-based 安全限制的混合方案。

### 为什么评估指标是 `f1_at_threshold` 而不是 `f1_macro`？

- `f1_macro` 使用默认阈值 0.5，对类别不平衡不友好
- `f1_at_threshold` 搜索最优阈值后计算，更能反映模型的真实区分能力
- 在实际部署时也会使用优化后的阈值

### 与原版 autoresearch 的区别

| 方面 | 原版 (LLM 预训练) | GBP-Cascade (医学图像分类) |
|---|---|---|
| 任务 | 语言模型预训练 | 胆囊息肉二分类 |
| 指标 | `val_bpb` (越低越好) | `f1_at_threshold` (越高越好) |
| 训练模式 | 固定 5 分钟时间 | 固定 epoch + 时间安全限 |
| 模型 | 从零训练 GPT | timm 预训练微调 |
| 数据 | 大规模文本 | 1229 张医学图像 |
| 评估 | 训练后评估一次 | 每 N epoch 评估 + 保存最优 |
| 过拟合 | 不太可能 | **核心挑战** |
| 包管理 | uv | pip |

---

## 常见问题

### Q: Agent 运行一晚上大概能跑多少实验？

每个实验 ~5-8 分钟 (100 epochs + 评估)。8 小时约 **60-100 个实验**。

### Q: 如果 Agent 把代码改坏了怎么办？

Git 分支保护了主代码。所有实验在独立分支 `autoresearch/<tag>` 上进行。最差情况只需 `git checkout main`。

### Q: 我能同时在多个 GPU 上跑吗？

可以。创建不同的分支 (如 `autoresearch/mar24-gpu0`, `autoresearch/mar24-gpu1`)，在不同 GPU 上分别启动 Agent。注意设置 `CUDA_VISIBLE_DEVICES`：

```bash
# GPU 0
CUDA_VISIBLE_DEVICES=0 python train_task2.py > run.log 2>&1

# GPU 1 (另一个终端)
CUDA_VISIBLE_DEVICES=1 python train_task2.py > run.log 2>&1
```

### Q: 如何让 Agent 探索我特别想尝试的方向？

修改 `program_gbp.md` 中的 "High-priority search directions" 部分。例如：

```markdown
### 本次重点方向
1. 只尝试 EfficientNet 系列模型 (b0, b1, b2, b3, b4)
2. 每个模型固定用 strong aug + balanced sampler
3. 重点调 learning rate 和 dropout
```

### Q: 如何在实验结束后集成多个模型？

AutoResearch 专注于单模型优化。集成可以事后进行：

```python
# 加载 top-3 模型的 checkpoint
# 对测试集取平均 softmax
# 搜索集成阈值
```

父项目 `GBP-Cascade/` 中已有 `20260323_task2_ensemble_eval.py` 可参考。

### Q: 如何恢复被 Agent 丢弃的实验？

所有 commit (包括被 `git reset` 的) 都可以通过 `git reflog` 找回：

```bash
git reflog
# 找到被丢弃的 commit hash
git checkout <commit-hash> -- train_task2.py
```

### Q: 我可以修改 prepare_task2.py 吗？

**不建议在 Agent 实验期间修改。** 但如果你需要：
- 添加新的 Transform 类
- 添加新的数据集类
- 修改评估逻辑

请在主分支上修改并提交，然后新建一个 autoresearch 分支。注意这会改变评估标准，之前的 results.tsv 不再可比。

---

## 致谢

- [Andrej Karpathy](https://github.com/karpathy/autoresearch) — autoresearch 原始理念和框架设计
- [timm](https://github.com/huggingface/pytorch-image-models) — 预训练图像模型库
- GBP-Cascade 项目前期 11 个手工实验积累的经验
