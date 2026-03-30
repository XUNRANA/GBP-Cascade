# JSON 标注 Label 说明

## 标注中的 5 种 Label

| Label | 形状类型 | 含义 |
|-------|---------|------|
| `gallbladder` | **矩形框** (rectangle) | 胆囊区域，每张图都有 |
| `gallbladder polyp` | **多边形** (polygon) | 胆囊息肉 |
| `pred` | **多边形** (polygon) | 病灶区域（医生标注） |
| `gallbladder adenoma` | **多边形** (polygon) | 腺瘤 |
| `gallbladder tubular adenoma` | **多边形** (polygon) | 管状腺瘤 |

## 代码中的使用方式

核心在 `task2_json_utils.py` 的两个函数：

### 函数 1：`get_gallbladder_rect()` — 只找 `gallbladder`

```
label == "gallbladder" 且 shape_type == "rectangle"
→ 提取矩形框坐标 [x1, y1, x2, y2]
→ 用于 ROI 裁剪
```

### 函数 2：`generate_lesion_mask()` — 找除了 gallbladder 以外的所有多边形

```
label != "gallbladder" 且 shape_type == "polygon"
→ 不管是 polyp、pred、adenoma 还是 tubular adenoma
→ 全部画到同一张二值 mask 上（有病灶=白，无=黑）
→ 用作第 4 通道输入
```

### 整体结构

```
JSON shapes
  ├── gallbladder (矩形) ──→ 裁剪用，告诉模型"胆囊在哪"
  └── 其他所有 (多边形) ──→ 生成 mask，告诉模型"病灶在哪"
         ├── gallbladder polyp
         ├── pred
         ├── gallbladder adenoma
         └── gallbladder tubular adenoma
```

**注意**：代码并不区分具体是哪种病灶，所有病灶多边形都被统一合并成一张二值 mask。模型看到的只是"这里有/没有病灶"，而不是"这里是息肉还是腺瘤"。
