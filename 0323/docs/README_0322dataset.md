# 0322dataset 数据集说明

## 一、数据集概述

胆囊超声影像二分类数据集（Task 2），用于区分 **良性肿瘤 (benign)** 与 **无肿瘤 (no_tumor)**。
每张图片均配有 LabelMe 格式的 JSON 标注文件，标注由医生完成。

## 二、目录结构

```
0322dataset/
├── benign/             # 良性肿瘤 (label=0), 438 张
│   ├── xxx.png
│   └── xxx.json
├── no_tumor/           # 无肿瘤 (label=1), 1400 张
│   ├── xxx.png
│   └── xxx.json
├── malignant/          # 恶性肿瘤, 386 张 (不参与 Task 2)
│   ├── xxx.png
│   └── xxx.json
├── task_2_train.xlsx   # 训练集划分 (1229 张: benign 309 + no_tumor 920)
├── task_2_test.xlsx    # 测试集划分 (523 张: benign 129 + no_tumor 394)
└── 划分报告.txt
```

**Task 2 二分类定义：label=0 (benign) vs label=1 (no_tumor)**

## 三、标注标签说明

JSON 文件遵循 LabelMe 格式，`shapes` 字段包含以下标签：

| 标签 | 形状类型 | 含义 | 说明 |
|------|----------|------|------|
| `gallbladder` | rectangle | 胆囊 ROI 区域 | **所有图片都有**。在 malignant 中覆盖范围大于胆囊本体，包含周边病灶组织 |
| `gallbladder polyp` | polygon | 胆囊息肉病灶 | 医生标注的息肉轮廓 |
| `pred` | polygon | 病灶区域 | 医生标注的病灶区域（非模型预测，是医生标注名） |
| `gallbladder tubular adenoma` | polygon | 胆囊管状腺瘤 | 腺瘤亚型病灶轮廓 |
| `gallbladder  adenoma` | polygon | 胆囊腺瘤 | 腺瘤病灶轮廓（注意标签中有两个空格） |

> **核心区分**：`gallbladder` 是定位框（矩形），其余标签均为病灶区域标注（多边形）。

## 四、各类别标注分布

### benign（438 张）
| 标签组合 | 数量 |
|----------|------|
| gallbladder + gallbladder polyp | 226 |
| gallbladder + pred | 151 |
| gallbladder + gallbladder tubular adenoma | 31 |
| gallbladder + gallbladder adenoma | 28 |
| gallbladder + gallbladder adenoma + gallbladder polyp | 2 |

### no_tumor（1400 张）
| 标签组合 | 数量 |
|----------|------|
| gallbladder + gallbladder polyp | 783 |
| gallbladder + pred | 612 |
| gallbladder + gallbladder adenoma | 2 |
| gallbladder + gallbladder polyp + gallbladder tubular adenoma | 1 |
| gallbladder + gallbladder tubular adenoma | 1 |
| 仅 gallbladder（无病灶标注） | 1 |

### malignant（386 张，不参与 Task 2）
| 标签组合 | 数量 |
|----------|------|
| 仅 gallbladder | 386 |

## 五、数据特点

1. **类别不平衡**：no_tumor : benign ≈ 3:1（训练集 920:309）
2. **两类都有病灶标注**：benign 和 no_tumor 均有息肉/腺瘤等标注，区分难度较高
3. **malignant 无病灶多边形**：仅有扩大的胆囊矩形框，覆盖周边侵犯组织
4. **图片尺寸**：320×320 像素

## 六、Task 2 可利用的标注信息分析

### 方案 1：仅用 `gallbladder` ROI 裁剪
- 用矩形框裁剪出胆囊区域，去除无关背景
- 让模型聚焦胆囊内部纹理差异
- **优势**：简单有效，减少噪声；所有图片都可用

### 方案 2：病灶 mask 作为额外输入通道
- 将 `gallbladder polyp` / `pred` / `adenoma` 等多边形生成二值 mask
- 输入变为 4 通道：RGB + lesion_mask
- **优势**：显式引入病灶位置先验，帮助模型关注关键区域
- **注意**：两类都有病灶标注，模型需学习的是病灶形态/纹理差异而非有无

### 方案 3：Dual-branch（双分支）
- 分支 A：全图或 gallbladder ROI 裁剪 → 全局特征
- 分支 B：病灶区域裁剪 → 局部细节特征
- 融合后分类
- **优势**：兼顾全局上下文和病灶细节

### 方案 4：病灶属性作为辅助特征
- 从标注提取：病灶面积、面积占比（病灶/胆囊）、病灶数量、病灶位置
- 作为辅助输入拼接到 CNN 特征后
- **优势**：利用结构化先验知识

### 推荐策略
**方案 1 + 方案 2 组合**：先用 gallbladder ROI 裁剪聚焦胆囊区域，再叠加病灶 mask 通道。实现简单、信息利用充分，是最具性价比的方案。
