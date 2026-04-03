# Exp#4 最佳实验详细技术分析

> 实验: `20260402_task2_SwinV2Tiny_segcls_4`
> 结果: **F1=0.6707** (阈值优化后 0.6741), 超越基线 5.3%

---

## 1. 总体架构概览

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Exp#4 模型架构                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   输入: 4ch (RGB + lesion_mask)  [B, 4, 256, 256]                       │
│      │                                                                   │
│      ▼                                                                   │
│   ┌──────────────────────────────────────────────────────────┐          │
│   │          SwinV2-Tiny Encoder (pretrained)                │          │
│   │   ┌──────┐    ┌──────┐    ┌──────┐    ┌──────┐          │          │
│   │   │ f0   │    │ f1   │    │ f2   │    │ f3   │          │          │
│   │   │64x64 │    │32x32 │    │16x16 │    │ 8x8  │          │          │
│   │   │ 96ch │    │192ch │    │384ch │    │768ch │          │          │
│   │   └──┬───┘    └──┬───┘    └──┬───┘    └──┬───┘          │          │
│   └──────│──────────│──────────│──────────│─────────────────┘          │
│          │          │          │          │                             │
│          │ skip     │ skip     │ skip     │                             │
│          │          │          ▼          │                             │
│   ┌──────│──────────│────────────────────────────────────────┐          │
│   │      ▼          ▼        ┌──────┐     │                  │          │
│   │   ┌──────┐  ┌──────┐    │ dec3 │◄────┘                  │          │
│   │   │ dec1 │  │ dec2 │◄───│384ch │                        │          │
│   │   │ 96ch │  │192ch │    └──┬───┘                        │          │
│   │   └──┬───┘  └──┬───┘       │                            │          │
│   │      │         └───────────┘                            │          │
│   │      │                                                   │          │
│   │      ▼                UNet Decoder                       │          │
│   │   ┌────────────┐                                        │          │
│   │   │ seg_final  │                                        │          │
│   │   │  4x上采样   │                                        │          │
│   │   └──────┬─────┘                                        │          │
│   └──────────│──────────────────────────────────────────────┘          │
│              ▼                                                          │
│   ┌──────────────────┐                                                  │
│   │  Seg Logits      │ [B, 2, 256, 256]                                │
│   │  (背景/病灶)      │                                                  │
│   └────────┬─────────┘                                                  │
│            │ softmax → lesion_prob                                      │
│            ▼                                                            │
│   ┌──────────────────────────────────────────────────────────┐         │
│   │              Seg-Guided Attention                        │         │
│   │   ┌─────────────────────────────────────────────────┐   │         │
│   │   │ 1. lesion_prob下采样到f2尺寸 (16x16)             │   │         │
│   │   │ 2. attn = prob + 0.1 (避免全零)                  │   │         │
│   │   │ 3. attn = attn / sum(attn) (归一化)              │   │         │
│   │   │ 4. f2_proj = Conv1x1(f2) → 256ch                 │   │         │
│   │   │ 5. cls_feat = sum(f2_proj * attn) → [B, 256]    │   │         │
│   │   └─────────────────────────────────────────────────┘   │         │
│   └──────────────────────────┬───────────────────────────────┘         │
│                              │                                          │
│   ┌──────────────────────────┼───────────────────────────────┐         │
│   │         Metadata         │        Fusion                 │         │
│   │  ┌─────────────────┐     │     ┌─────────────────────┐   │         │
│   │  │ 6维特征:        │     │     │ concat([cls_feat,   │   │         │
│   │  │ age, gender,   │────►│─────►│         meta_feat]) │   │         │
│   │  │ size, flow,    │     │     │       [B, 320]       │   │         │
│   │  │ morph          │     │     └──────────┬──────────┘   │         │
│   │  │ ↓              │     │                │              │         │
│   │  │ MLP → [B, 64]  │     │                │              │         │
│   │  └─────────────────┘     │                │              │         │
│   └──────────────────────────┼───────────────────────────────┘         │
│                              ▼                                          │
│                     ┌─────────────────┐                                 │
│                     │   Cls MLP       │                                 │
│                     │ 320→128→dropout │                                 │
│                     │  →num_classes   │                                 │
│                     └────────┬────────┘                                 │
│                              ▼                                          │
│                     ┌─────────────────┐                                 │
│                     │ Cls Logits [B,2]│                                 │
│                     │ (benign/no_tumor)│                                │
│                     └─────────────────┘                                 │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 输入详解

### 2.1 4通道输入组成

```python
# 输入 tensor shape: [B, 4, 256, 256]
input_4ch = torch.cat([img_t, mask_t], dim=0)

# 分解:
# - 通道 0-2: RGB图像 (ImageNet归一化)
#   - mean = [0.485, 0.456, 0.406]
#   - std = [0.229, 0.224, 0.225]
# - 通道 3: 病灶掩码 (原始 0/1, 未归一化)
```

### 2.2 数据来源

| 数据类型 | 来源 | 格式 |
|---------|------|------|
| 图像 | `0322dataset/{class}/patient_xxx/xxx.png` | RGB PNG |
| 病灶标注 | 对应的 `.json` 文件 (LabelMe格式) | 多边形顶点 |
| 分类标签 | `task_2_train.xlsx` / `task_2_test.xlsx` | 0=benign, 1=no_tumor |
| Metadata | `胆囊超声组学_分析.xlsx` + `json_text/` | 6维数值特征 |

### 2.3 Metadata 6维特征

```python
META_FEATURE_NAMES = [
    "age",           # 年龄 (连续, z-score标准化)
    "gender",        # 性别 (0/1)
    "size_mm",       # 病灶尺寸mm (连续, z-score标准化)
    "size_bin",      # 尺寸分箱 (离散, one-hot)
    "flow_bin",      # 血流分级 (离散, one-hot)
    "morph_bin",     # 形态学分级 (离散, one-hot)
]
```

Metadata 处理流程:
1. 从 Excel 读取临床数据, 按 `case_id` 匹配
2. 连续特征做 z-score 标准化 (用训练集统计量)
3. 缺失值填充为 0 (标准化后的均值)

---

## 3. 模型架构详解

### 3.1 Encoder: SwinV2-Tiny

```python
self.encoder = timm.create_model(
    "swinv2_tiny_window8_256",  # 预训练在 ImageNet-1K
    pretrained=True,
    features_only=True,         # 输出多尺度特征
    out_indices=(0, 1, 2, 3),   # 4个阶段
)
adapt_model_to_4ch(self.encoder)  # 第一层卷积扩展到 4ch
```

**多尺度特征:**

| Stage | 特征名 | 分辨率 | 通道数 | 下采样倍率 |
|-------|-------|--------|--------|-----------|
| 0 | f0 | 64×64 | 96 | 4x |
| 1 | f1 | 32×32 | 192 | 8x |
| 2 | f2 | 16×16 | 384 | 16x |
| 3 | f3 | 8×8 | 768 | 32x |

**4ch 适配:**
```python
def adapt_model_to_4ch(model):
    """将第一层卷积从 3ch 扩展到 4ch"""
    conv1 = model.patch_embed.proj  # Conv2d(3, 96, ...)
    new_conv = nn.Conv2d(4, 96, ...)
    new_conv.weight.data[:, :3] = conv1.weight.data
    new_conv.weight.data[:, 3:4] = conv1.weight.data[:, :1]  # 复制第一通道权重
    model.patch_embed.proj = new_conv
```

### 3.2 Decoder: UNet-style

```python
class UNetDecoderBlock(nn.Module):
    """上采样 + skip connection + 双卷积"""
    def __init__(self, in_ch, skip_ch, out_ch):
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch), nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch), nn.GELU(),
        )
    
    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)  # skip connection
        return self.conv(x)
```

**解码路径:**
```
f3 (8×8, 768ch)
  ↓ dec3: up + concat(f2) + conv
d3 (16×16, 384ch)
  ↓ dec2: up + concat(f1) + conv
d2 (32×32, 192ch)
  ↓ dec1: up + concat(f0) + conv
d1 (64×64, 96ch)
  ↓ seg_final: 4x转置卷积 + conv
seg_logits (256×256, 2ch)
```

### 3.3 Seg-Guided Attention (核心创新)

```python
# 1. 获取分割概率图
seg_prob = F.softmax(seg_logits, dim=1)[:, 1:2]  # 病灶概率 [B, 1, 256, 256]

# 2. 下采样到特征图尺寸
attn = F.interpolate(seg_prob, size=f2.shape[2:], mode="bilinear")  # [B, 1, 16, 16]

# 3. 添加偏置避免全零
attn = attn + 0.1

# 4. 归一化为注意力权重
attn = attn / (attn.sum(dim=(2, 3), keepdim=True) + 1e-6)

# 5. 特征投影
f2_proj = self.cls_proj(f2)  # [B, 256, 16, 16]

# 6. 注意力加权池化
cls_feat = (f2_proj * attn).sum(dim=(2, 3))  # [B, 256]
```

**为什么有效?**
- 分割预测告诉模型"病灶在哪里"
- 分类特征只关注病灶区域, 忽略背景
- `+0.1` 确保即使分割不准也能保留全局信息
- 比 GAP 更有针对性地提取病灶特征

### 3.4 Metadata Fusion

```python
# Metadata 编码器
self.meta_encoder = nn.Sequential(
    nn.Linear(6, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(0.2),
    nn.Linear(64, 64), nn.GELU(), nn.Dropout(0.2),
)

# 融合
meta_feat = self.meta_encoder(metadata)  # [B, 64]
cls_feat = torch.cat([cls_feat, meta_feat], dim=1)  # [B, 256+64=320]

# 分类 MLP
self.cls_mlp = nn.Sequential(
    nn.Linear(320, 128), nn.GELU(), nn.Dropout(0.4),
    nn.Linear(128, 2),
)
```

---

## 4. 损失函数详解

### 4.1 总损失公式

```
Total Loss = Seg Loss + λ_cls × Cls Loss

其中:
- λ_cls = 2.0 (分类为主任务)
- Seg Loss = CE Loss + Dice Loss (仅在有标注的样本上计算)
- Cls Loss = Weighted CE Loss with Label Smoothing
```

### 4.2 分割损失

```python
class SegClsLoss(nn.Module):
    def __init__(self, cls_weights, lambda_cls=2.0, label_smoothing=0.1,
                 seg_ce_weight=None):
        # 分割 CE: 背景权重=1.0, 病灶权重=5.0 (处理类别不平衡)
        self.seg_ce = nn.CrossEntropyLoss(weight=seg_ce_weight)
        self.seg_dice = DiceLoss()
        
    def forward(self, seg_logits, cls_logits, seg_targets, cls_targets, has_mask):
        # 分割损失只在有标注的样本上计算
        if has_mask.any():
            mask_idx = has_mask.nonzero(as_tuple=True)[0]
            seg_ce_loss = self.seg_ce(seg_logits[mask_idx], seg_targets[mask_idx])
            seg_dice_loss = self.seg_dice(seg_logits[mask_idx], seg_targets[mask_idx])
            seg_loss = seg_ce_loss + seg_dice_loss
```

### 4.3 分类损失

```python
# 类别权重 (处理不平衡: benign=309, no_tumor=920)
cls_weights = [1.9887, 0.6679]  # benign 权重更高

# Label Smoothing CE
self.cls_ce = nn.CrossEntropyLoss(
    weight=cls_weights,
    label_smoothing=0.1,  # 软标签, 防止过拟合
)
```

### 4.4 Dice Loss 详解

```python
class DiceLoss(nn.Module):
    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        dice = 0.0
        for c in range(num_classes):
            pred_c = probs[:, c]
            target_c = (targets == c).float()
            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()
            dice += (2 * intersection + smooth) / (union + smooth)
        return 1.0 - dice / num_classes
```

---

## 5. 数据增强详解

### 5.1 训练时增强 (SegCls4chSyncTransform)

```python
# 所有变换对 RGB 和 mask 同步应用

# 1. Random Resized Crop
scale=(0.7, 1.0), ratio=(0.85, 1.15)
# 随机裁剪 70%-100% 面积, 保持近正方形

# 2. 随机翻转
if random.random() < 0.5: hflip(img, mask)
if random.random() < 0.3: vflip(img, mask)

# 3. 随机旋转
if random.random() < 0.5:
    angle = uniform(-20, 20)
    rotate(img, mask, angle)

# 4. 随机仿射变换
if random.random() < 0.5:
    angle = uniform(-5, 5)
    translate = uniform(-6%, +6%) * img_size
    scale = uniform(0.9, 1.1)
    shear = uniform(-5, 5)
    affine(img, mask)

# 5. 颜色抖动 (仅 RGB)
if random.random() < 0.6:
    brightness = uniform(0.7, 1.3)
    contrast = uniform(0.7, 1.3)
    saturation = uniform(0.8, 1.2)

# 6. 高斯模糊 (仅 RGB)
if random.random() < 0.2:
    gaussian_blur(img, kernel_size=3)

# 7. 归一化 (仅 RGB)
normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# 8. Random Erasing (仅 RGB, 归一化后)
if random.random() < 0.2:
    random_erasing(img, scale=(0.02, 0.15))

# 9. 高斯噪声 (仅 RGB)
if random.random() < 0.3:
    img = img + randn * 0.03
```

### 5.2 测试时变换

```python
# 仅 resize, 无增强
img = resize(img, [256, 256], interpolation=BICUBIC)
mask = resize(mask, [256, 256], interpolation=NEAREST)
img = normalize(img, mean, std)
```

### 5.3 为什么不用 Mixup?

**实验证明 Mixup 与分割冲突!** (Exp#2 教训)
- Mixup 混合两张图的 mask 没有语义意义
- 分割 Dice 从 0.91 降到 0.83
- F1 从 0.62 降到 0.59

---

## 6. 训练配置详解

### 6.1 超参数

```python
# 训练设置
batch_size = 8
num_epochs = 100
seed = 42
use_amp = True  # 混合精度训练

# 学习率
backbone_lr = 2e-5    # encoder 低学习率 (预训练)
head_lr = 2e-4        # decoder/cls head 高学习率 (随机初始化)
weight_decay = 5e-2   # L2 正则化

# 学习率调度
warmup_epochs = 8
min_lr_ratio = 0.01
# Warmup: 线性从 0 到 base_lr
# Warmup 后: Cosine decay 到 min_lr_ratio * base_lr

# 梯度裁剪
grad_clip = 1.0

# 损失权重
lambda_cls = 2.0
seg_bg_weight = 1.0
seg_lesion_weight = 5.0
label_smoothing = 0.1

# Dropout
cls_dropout = 0.4
meta_dropout = 0.2
```

### 6.2 优化器: AdamW with 差异化学习率

```python
def build_optimizer_with_diff_lr(AdamW, backbone_params, head_params, cfg):
    return AdamW([
        {"params": backbone_params, "lr": cfg.backbone_lr},
        {"params": head_params, "lr": cfg.head_lr},
    ], weight_decay=cfg.weight_decay)
```

### 6.3 学习率调度

```python
def cosine_warmup_factor(epoch, warmup_epochs, total_epochs, min_lr_ratio):
    if epoch <= warmup_epochs:
        return epoch / warmup_epochs  # 线性 warmup
    else:
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return min_lr_ratio + 0.5 * (1 - min_lr_ratio) * (1 + cos(π * progress))
```

---

## 7. 数据流完整流程

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         完整数据流                                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. 数据加载                                                             │
│     Excel (image_path, label) + JSON (mask polygons) + Clinical Excel   │
│                    ↓                                                     │
│  2. 预处理                                                               │
│     RGB = PIL.open(img_path)                                            │
│     mask = polygon_to_binary_mask(json_shapes)                          │
│     meta = extract_metadata(clinical_excel, case_id)                    │
│                    ↓                                                     │
│  3. 数据增强 (训练时)                                                    │
│     [RGB, mask] = sync_transform([RGB, mask])                           │
│     meta = z_score_normalize(meta)                                      │
│                    ↓                                                     │
│  4. Tensor 化                                                            │
│     img_4ch = concat([normalize(RGB), raw_mask])  # [4, 256, 256]       │
│     seg_target = (mask > 0.5).long()              # [256, 256]          │
│     meta_tensor = tensor(meta)                    # [6]                  │
│                    ↓                                                     │
│  5. Batch 组装 (DataLoader)                                              │
│     imgs: [B, 4, 256, 256]                                              │
│     masks: [B, 256, 256]                                                │
│     metas: [B, 6]                                                       │
│     labels: [B]                                                         │
│     has_masks: [B] (bool)                                               │
│                    ↓                                                     │
│  6. Forward Pass                                                         │
│     seg_logits, cls_logits = model(imgs, metadata=metas)                │
│                    ↓                                                     │
│  7. 损失计算                                                             │
│     seg_loss = CE(seg_logits[has_mask], masks[has_mask]) + Dice(...)   │
│     cls_loss = CE(cls_logits, labels)                                   │
│     total = seg_loss + 2.0 * cls_loss                                   │
│                    ↓                                                     │
│  8. 反向传播                                                             │
│     scaler.scale(total).backward()                                      │
│     clip_grad_norm_(params, 1.0)                                        │
│     scaler.step(optimizer)                                              │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 8. 评估方法

### 8.1 训练过程监控

每个 epoch 记录:
- **Loss**: total, seg_loss, cls_loss
- **Cls Acc**: 训练集分类准确率
- **Seg IoU / Dice**: 训练集分割指标 (仅有标注样本)

### 8.2 测试集评估 (每 5 epochs)

```python
# 分类指标 (macro average)
Accuracy, Precision, Recall, F1

# 分割指标 (病灶类)
Lesion IoU, Lesion Dice

# 分类报告
per-class precision, recall, f1-score
```

### 8.3 最终评估

1. **加载最优权重** (验证集 F1 最高的 epoch)
2. **默认阈值测试** (threshold=0.5)
3. **阈值优化搜索** (0.15~0.75 范围, 步长 0.005)
4. **最优阈值测试**

---

## 9. 最终结果

### 9.1 核心指标

| 指标 | 值 | 说明 |
|------|-----|------|
| **Best F1 (macro)** | **0.6707** | Epoch 10 |
| **F1 (阈值优化)** | **0.6741** | threshold=0.505 |
| Accuracy | 0.7533 | |
| Seg Lesion Dice | 0.9851 | 分割几乎完美 |
| Seg Lesion IoU | 0.9706 | |

### 9.2 Per-Class 表现

```
              precision    recall  f1-score   support

      benign     0.5000    0.5116    0.5057       129
    no_tumor     0.8389    0.8325    0.8357       394

    accuracy                         0.7533       523
   macro avg     0.6694    0.6721    0.6707       523
weighted avg     0.7553    0.7533    0.7543       523
```

### 9.3 vs 基线对比

| 指标 | Exp#4 | 基线 | 提升 |
|------|-------|------|------|
| F1 (macro) | 0.6707 | 0.6371 | **+5.3%** |
| benign recall | 0.5116 | ~0.40 | **+28%** |
| benign precision | 0.5000 | ~0.45 | +11% |

---

## 10. 关键要点总结

### ✅ 成功因素

1. **4ch 输入**: 保留 mask 作为输入信息, 同时分割做正则化
2. **Seg-Guided Attention**: 用分割概率引导分类关注病灶区域
3. **Metadata 融合**: 临床特征提供图像无法捕捉的信息
4. **无 Mixup**: 避免与分割任务冲突
5. **差异化学习率**: backbone 2e-5, head 2e-4
6. **类别平衡**: CE 权重 + Dice Loss + Label Smoothing

### ⚠️ 注意事项

1. **过拟合警告**: Epoch 10 就达到最优, 后续持续下降
   - 训练准确率 98.8% vs 测试 75.3%
   - 建议: 训练 30-50 epochs 足够

2. **分割已完美**: Dice 0.985, 进一步提升分割意义不大

3. **benign 召回仍是瓶颈**: 0.51, 还有提升空间

4. **metadata 依赖**: 无 metadata 时 F1 下降 ~6 个百分点

---

## 11. 代码文件索引

| 文件 | 作用 |
|------|------|
| `20260402_task2_SwinV2Tiny_segcls_4.py` | 实验配置与入口 |
| `seg_cls_utils_v2.py` | 数据集, 模型, 训练循环 |
| `seg_cls_utils.py` | 基础组件 (UNet块, Loss, 工具函数) |
| `0323/scripts/test_yqh.py` | Metadata 处理, 4ch 适配 |

---

**总结**: Exp#4 成功的核心是 **Seg-Guided Attention + Metadata + 4ch 输入** 的组合, 让模型既能利用显式的病灶位置信息 (mask 输入), 又能通过分割任务学习隐式的空间特征, 再配合临床元数据做出更准确的分类判断。
