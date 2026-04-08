# 脚本详细解析：20260401_task2_SwinV2Tiny_full4ch_strongaug_7_text_13.py

## 一、总体结论：该脚本并未使用超声报告文本信息

**尽管文件名包含 `_text_13` 后缀，但该脚本实际上完全没有使用超声报告的文本信息（`text_bert` 字段）。** 它只是从 JSON 文件中提取了少量**结构化数值特征**（`size_mm`, `size_bin`, `flow_bin`, `morph_bin`），并非真正的文本语义信息。

---

## 二、脚本架构概览

```
┌─────────────────────────────────────────────────────────┐
│               主脚本 (Exp#7, _text_13.py)               │
│                                                         │
│  ┌─────────┐    ┌──────────────────┐    ┌───────────┐  │
│  │  Config  │    │ SwinV2TinyMeta   │    │ build_*() │  │
│  │  配置类  │    │ Fusion 模型类    │    │ 构建函数  │  │
│  └─────────┘    └──────────────────┘    └───────────┘  │
│                          ↓                              │
│              调用 run_experiment()                       │
└─────────────────────────────────────────────────────────┘
                          ↓ import
┌─────────────────────────────────────────────────────────┐
│               工具库 (test_yqh.py)                       │
│                                                         │
│  - GBPDatasetFull4chWithMeta   数据集类                  │
│  - StrongSyncTransform         强增强变换                │
│  - build_case_meta_table()     元数据加载                │
│  - run_experiment()            训练循环                   │
│  - mixup_cutmix_data()         数据混合                  │
│  - ...                                                   │
└─────────────────────────────────────────────────────────┘
```

---

## 三、信息来源与使用方式详解

### 3.1 数据来源

脚本使用了**三个数据源**：

| 数据源 | 路径 | 提取的信息 | 用途 |
|--------|------|-----------|------|
| **超声图像 + LabelMe标注** | `0322dataset/{benign,no_tumor}/*.png` + `.json` | RGB图像 + 病灶多边形mask | 4通道图像输入 |
| **临床Excel表** | `胆囊超声组学_分析.xlsx` | `age`(年龄), `gender`(性别) | 元数据分支的2个特征 |
| **JSON特征文件** | `json_text/*.json` | `size_mm`, `size_bin`, `flow_bin`, `morph_bin` | 元数据分支的4个特征 |

### 3.2 JSON 文件结构

以 `json_text/00166451.json` 为例：

```json
{
    "case_id": "00166451",
    "text_bert": "胆囊：胆囊轮廓不清楚，胆囊壁显示不清，几乎未见胆汁回声；
                  胆囊区域几乎充满实性回声，范围约73mm×40mm，形态欠规则，
                  边缘不清楚，内部回声不均匀...",
    "feat": {
        "size_mm": 40.0,
        "size_bin": 2,
        "flow_bin": 0,
        "morph_bin": 2,
        "mask": { "size": 1, "flow": 1, "morph": 1 }
    }
}
```

**关键区分：**
- `text_bert` — 超声报告原始文本（医生的完整描述性诊断）→ **本脚本完全未使用**
- `feat` — 预提取的结构化数值特征 → **本脚本使用了其中4个字段**

### 3.3 JSON 文件中文本信息的加载路径

在 `test_yqh.py` 的 `load_json_meta_table()` 函数（第204-232行）中：

```python
def load_json_meta_table(json_feature_root):
    rows = []
    for fp in Path(json_feature_root).glob("*.json"):
        data = load_annotation(fp)
        feat = data.get("feat", {})    # ← 只取 feat 字典
        rows.append({
            "case_id_norm": normalize_case_id(fp.stem),
            "size_mm":   _to_float(feat.get("size_mm")),    # 病灶尺寸(mm)
            "size_bin":  _to_float(feat.get("size_bin")),    # 尺寸分档(0/1/2)
            "flow_bin":  _to_float(feat.get("flow_bin")),    # 血流分档(0/1/2)
            "morph_bin": _to_float(feat.get("morph_bin")),   # 形态分档(0/1/2)
        })
    return pd.DataFrame(rows)
```

可以清楚看到：
- 只读取了 `data["feat"]` 中的 4 个数值字段
- **完全忽略了 `data["text_bert"]`**，没有任何代码读取该字段

---

## 四、实际使用的6维元数据（Metadata）详解

### 4.1 特征定义

```python
# test_yqh.py 第114行
META_FEATURE_NAMES = ["age", "gender", "size_mm", "size_bin", "flow_bin", "morph_bin"]
```

| 特征 | 来源 | 类型 | 含义 |
|------|------|------|------|
| `age` | Excel `年龄`列 | 连续值 | 患者年龄 |
| `gender` | Excel `性别`列 | 二值 (0/1) | 男=1, 女=0 |
| `size_mm` | JSON `feat.size_mm` | 连续值 | 病灶超声测量尺寸(mm) |
| `size_bin` | JSON `feat.size_bin` | 离散 (0/1/2) | 尺寸分档: 0=小, 1=中, 2=大 |
| `flow_bin` | JSON `feat.flow_bin` | 离散 (0/1/2) | 血流分档: 0=无, 1=少, 2=丰富 |
| `morph_bin` | JSON `feat.morph_bin` | 离散 (0/1/2) | 形态分档: 0=规则, 1=较规则, 2=不规则 |

### 4.2 特征预处理流程

```
原始值 → 缺失值填充(训练集中位数) → Z-score标准化 → 6维向量
```

具体代码（`test_yqh.py`）：

```python
# 1. 拟合统计量 (仅训练集)
def fit_meta_stats(df, feature_names):
    for col in feature_names:
        fill = median(valid_values)     # 中位数填充缺失值
        mean = filled_series.mean()     # 均值
        std  = filled_series.std()      # 标准差
    return stats

# 2. 编码单个样本
def encode_meta_row(row, stats, feature_names):
    for col in feature_names:
        v = row[col] if not NaN else stats[col]["fill"]  # 缺失→中位数
        v = (v - mean) / std                              # Z-score标准化
    return torch.tensor(vals, dtype=torch.float32)        # → 6D tensor
```

---

## 五、模型架构详解

### 5.1 SwinV2TinyMetaFusion 模型

```python
class SwinV2TinyMetaFusion(nn.Module):
    def __init__(self, meta_dim=6, num_classes=2, meta_hidden_dim=64, ...):
        # 图像分支: SwinV2-Tiny 骨干网络 (ImageNet预训练)
        self.backbone = timm.create_model("swinv2_tiny_window8_256",
                                           pretrained=True, num_classes=0)
        adapt_model_to_4ch(self.backbone)  # 3ch → 4ch (第4通道=mask)

        # 元数据分支: 2层MLP
        self.meta_encoder = nn.Sequential(
            nn.Linear(6, 64),       # 6D → 64D
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 64),      # 64D → 64D
            nn.GELU(),
            nn.Dropout(0.2),
        )

        # 融合分类头: 拼接后MLP
        # SwinV2-Tiny的num_features = 768
        self.fusion_head = nn.Sequential(
            nn.Linear(768 + 64, 384),  # 832D → 384D
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(384, 2),          # 384D → 2类
        )
```

### 5.2 前向传播流程

```
输入图像 (4ch: RGB+mask)                      元数据 (6D向量)
        ↓                                           ↓
  SwinV2-Tiny Backbone                        Meta Encoder
  (4ch→768D 全局特征)                          (6D→64D)
        ↓                                           ↓
        img_feat (768D)                       meta_feat (64D)
                    ↘                       ↙
                      torch.cat → 832D
                            ↓
                     Fusion Head MLP
                      832D → 384D → 2D
                            ↓
                        分类输出
```

对应代码：

```python
def forward(self, x, metadata=None):
    # 1. 图像特征提取
    img_feat = self.backbone.forward_features(x)        # → [B, 768]
    img_feat = self.backbone.forward_head(img_feat, pre_logits=True)

    # 2. 元数据编码（缺失时用全零替代）
    if metadata is None:
        metadata = torch.zeros(x.size(0), self.meta_dim, ...)
    meta_feat = self.meta_encoder(metadata.float())     # → [B, 64]

    # 3. 拼接融合 + 分类
    fused = torch.cat([img_feat, meta_feat], dim=1)     # → [B, 832]
    return self.fusion_head(fused)                       # → [B, 2]
```

---

## 六、训练策略详解

### 6.1 数据增强 — StrongSyncTransform

图像和mask同步进行以下增强（仅训练时）：

| 增强方式 | 概率 | 参数 |
|----------|------|------|
| RandomResizedCrop | 100% | scale=(0.7,1.0), ratio=(0.85,1.15) |
| 水平翻转 | 50% | - |
| 垂直翻转 | 30% | - |
| 随机旋转 | 50% | ±20° |
| 随机仿射(shear+平移) | 50% | shear±5°, translate 6% |
| 颜色抖动(强) | 60% | brightness/contrast ±30%, saturation ±20% |
| 高斯模糊 | 20% | kernel=3 |
| Random Erasing | 20% | scale=(0.02,0.15) |
| 高斯噪声 | 30% | σ=0.03 |

### 6.2 Mixup / CutMix

训练时对每个batch随机选择 Mixup 或 CutMix（50/50概率）：

```
Mixup:   images_mixed = λ * img_A + (1-λ) * img_B       (λ ~ Beta(0.4, 0.4))
CutMix:  images_mixed = img_A 中挖掉一块，填入 img_B    (λ ~ Beta(1.0, 1.0))

元数据也同步混合: meta_mixed = λ * meta_A + (1-λ) * meta_B
标签也软化:       label_mixed = λ * onehot_A + (1-λ) * onehot_B
```

### 6.3 训练超参数

| 参数 | 值 |
|------|-----|
| 图像尺寸 | 256×256 |
| Batch Size | 8 |
| Epochs | 100 |
| Backbone LR | 2e-5 |
| Head LR | 2e-4 |
| Weight Decay | 0.05 |
| Warmup | 8 epochs |
| LR Schedule | Cosine with warmup |
| Label Smoothing | 0.1 |
| Grad Clip | 1.0 |
| AMP | True |
| 损失函数 | CrossEntropyLoss (类别加权 + Label Smoothing) |

### 6.4 差异化学习率

```python
# Backbone (SwinV2 预训练参数): lr = 2e-5  (慢速微调)
# Head (meta_encoder + fusion_head): lr = 2e-4 (快速学习)
```

---

## 七、信息利用方式总结

### 已使用的信息

```
┌──────────────────────────────────────────────────────────┐
│                    信息使用全景                            │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  图像分支 (4通道):                                        │
│    ├── Ch 1-3: RGB 超声图像原图                           │
│    └── Ch 4:   病灶mask (来自LabelMe标注JSON的polygon)    │
│                                                          │
│  元数据分支 (6维):                                        │
│    ├── age        ← Excel 年龄列                          │
│    ├── gender     ← Excel 性别列                          │
│    ├── size_mm    ← json_text/feat.size_mm               │
│    ├── size_bin   ← json_text/feat.size_bin              │
│    ├── flow_bin   ← json_text/feat.flow_bin              │
│    └── morph_bin  ← json_text/feat.morph_bin             │
│                                                          │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  ❌ 未使用的信息:                                         │
│    ├── text_bert: 超声报告原始文本 (json_text中)          │
│    ├── 回声类型、个数、壁厚 (Excel中)                     │
│    └── 形状、基底 (Excel中, 覆盖率低)                     │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### 关键区别："利用JSON文件" ≠ "利用文本信息"

| 概念 | 本脚本的做法 | 真正的文本利用 (如0408的Exp#16-18) |
|------|-------------|-----------------------------------|
| 数据源 | `json_text/*.json` 的 `feat` 字段 | `json_text/*.json` 的 `text_bert` 字段 |
| 提取内容 | 4个离散/连续数值 | 完整中文文本字符串 |
| 编码方式 | Z-score标准化 → 6D向量 | BERT tokenize → 768D语义向量 |
| 信息量 | 极低 (4个数字) | 极高 (平均130字的完整描述) |
| 模型需求 | 简单MLP | BERT预训练模型 (~110M参数) |
| 融合方式 | 拼接到图像特征后 | 拼接/交叉注意力/门控融合 |

---

## 八、与 0408 实验的对比

0408 的 Exp#16-18 才是真正利用文本信息的实验：

| 对比维度 | 本脚本 (Exp#7) | 0408 Exp#16 | 0408 Exp#17 | 0408 Exp#18 |
|----------|---------------|-------------|-------------|-------------|
| 文本使用 | ❌ 不使用 | ✅ BERT [CLS] | ✅ BERT 全token | ✅ BERT [CLS]+CrossAttn |
| 元数据 | 6D | 6D | 6D | 10D |
| 文本编码器 | 无 | BERT-base-chinese (冻结) | BERT-base-chinese (冻结) | BERT-base-chinese (冻结) |
| 文本融合 | 无 | 后期拼接 | 交叉注意力 | 门控三模态融合 |
| 分割辅助 | ❌ 无 | ✅ UNet + SegAttn | ✅ UNet + SegAttn | ✅ UNet + SegAttn |
| 骨干网络 | SwinV2-Tiny (timm) | SwinV2-Tiny (自定义) | SwinV2-Tiny (自定义) | SwinV2-Tiny (自定义) |

---

## 九、代码执行流程

```
main()
  └─→ run_experiment(cfg, build_model, build_dataloaders, build_optimizer, __file__)
        │
        ├─ 1. build_dataloaders(cfg)
        │     ├─ 加载 Excel → 图像路径 + 标签
        │     ├─ load_clinical_meta_table(Excel) → age, gender
        │     ├─ load_json_meta_table(json_text/) → size_mm, size_bin, flow_bin, morph_bin
        │     │   ⚠️ 此处只读 feat 字段，完全忽略 text_bert
        │     ├─ 合并 → 每个样本6维元数据
        │     ├─ fit_meta_stats(训练集) → 中位数、均值、标准差
        │     └─ 构建 DataLoader (train + test)
        │
        ├─ 2. build_model(cfg)
        │     └─ SwinV2TinyMetaFusion(meta_dim=6, num_classes=2)
        │
        ├─ 3. build_optimizer(model, cfg)
        │     └─ AdamW (backbone: 2e-5, head: 2e-4)
        │
        ├─ 4. 训练循环 (100 epochs)
        │     ├─ 每个batch: 图像增强 → Mixup/CutMix → 前向 → 反向
        │     ├─ 每5个epoch: 在测试集评估
        │     └─ 保存最优F1模型
        │
        └─ 5. 最终评估
              ├─ 加载最优权重 → 默认阈值评估
              └─ 阈值搜索 → 最优阈值评估
```
