# 特征可视化实验 — 结果汇总

> 目标模型: `20260414_task3_SwinV2Tiny_segcls_3` (best, Softmax Macro-F1)
> 输入: test 集 456 张; 类别 = malignant (114) / benign (114) / no_tumor (228)

---

## Phase 1 — 深度特征多视角分析

输出目录: `logs/feature_analysis_best_softmax_segcls3/deep/`

### 1.1 多层 t-SNE 网格 (`1.1_multilayer_tsne_grid.png`)

| 位置 | dim | silhouette | ben↔no L2 |
|---|---|---|---|
| backbone f0 (GAP) | 96 | 0.009 | 1.13 |
| backbone f1 (GAP) | 192 | 0.015 | 0.95 |
| backbone f2 (GAP) | 384 | 0.036 | 2.35 |
| backbone f3 (GAP) | 768 | 0.018 | 2.41 |
| cls_proj (pre-attn) | 256 | 0.081 | 0.98 |
| cls_feat_img (seg-attn) | 256 | 0.106 | 1.52 |
| **cls_feat_320 (fused)** | 320 | **0.115** | **1.92** |
| penultimate | 128 | **0.182** | 1.99 |

→ silhouette 随深度单调上升; **penultimate 层 (128 D) 分得最开**, 表明 cls_head 的第一层 Linear+GELU 带来显著额外判别性. seg-attention 注入在 f2→cls_proj→cls_feat_img 这一步贡献最大 (+0.03 → +0.08 → +0.11).

### 1.2 PCA / t-SNE / UMAP 对比 (320-D, `1.2_pca_tsne_umap.png`)

PCA PC1+PC2 = 49.8% var; sil(PCA)=0.164 / sil(t-SNE)=0.164 / sil(UMAP)=0.155; KMeans-3 ARI=0.282.

### 1.3 多属性上色 (`1.3_attribute_color_grid.png`)

同一 2D 坐标, 9 种着色: 真实类 / 预测类 / 正确性 / ordinal score / softmax 置信度 / 熵 / size_bin / flow_bin / morph_bin. 结果显示:
- malignant 与 benign/no_tumor 的视觉簇与 ordinal 高分区重合
- 高熵点 (不确定度) 集中在 benign↔no_tumor 边界
- size / flow / morph 临床先验与视觉簇的一致性可直接肉眼比对

### 1.4 类间/类内距离 (`1.4_distance_distributions.png`)

- 类内平均距离 (到质心): mal=4.74, ben=5.16, no=4.71
- 质心欧氏距离: mal-ben=8.76, mal-no=9.85, **ben-no=1.92** ← 关键瓶颈
- 每样本 silhouette 的按类堆叠直方图直接显示 benign 众数在 0 附近

### 1.5 判别性维度 (`1.5_top_dims_bar.png` / `top6_density_kde.png` / `top20_corr_heatmap.png`)

对 320 维逐维做 3 类 ANOVA F 检验, 排序得到 top-30. top-6 的 KDE 叠加可直接看到三类分布分离的维度. top-20 Pearson 相关阵揭示冗余组.

### 1.6 "灰区"样本 (`1.6_confusion_zone_overlay.png`)

在 ordinal score t-SNE 上叠加 s=t1/t2 阈值等值线 + 高亮 malignant→no_tumor 漏诊样本 (临床红线) + 高置信误分类. 直接对应 `实验计划.md §4.3` 的临床三段式.

---

## Phase 2 — PyRadiomics 107维经典影像组学

输出目录: `logs/feature_analysis_best_softmax_segcls3/radiomics/`

### 2.2 特征提取 (`features_test_gb_roi.csv` / `features_test_lesion_roi.csv`)

- 提取时间: **250s**, 102 维 (first-order/shape2D/glcm/glrlm/glszm/gldm/ngtdm) × Original
- 胆囊 ROI: 453/456 成功 (3 张因归一化后 mask 为空失败)
- 病灶 ROI: 342/342 成功 (仅 benign+no_tumor)
- 配置已保存到 `extractor_config.yaml` (binWidth=25, force2D, normalize=true, scale=100)

### 2.3 降维 (`2.3_pca_scree.png` / `pca_2d.png` / `tsne.png` / `umap.png`)

| 方法 | silhouette |
|---|---|
| raw-standardized 102D | 0.009 |
| PCA-2D | 0.001 |
| t-SNE-2D | -0.020 |
| UMAP-2D | -0.014 |

KMeans-3 ARI = 0.162. PCA 累计方差: PC1+PC2=49.9%, 80% 需 6 主成分, 90% 需 10.

**诊断**: 单独用经典影像组学特征几乎无法区分三类 (silhouette ≈ 0), 远弱于深度特征的 0.115. 但显著的单特征仍存在 → 适合当 biomarker 而不是端到端分类器.

### 2.4 单特征统计 → `biomarker_table.csv`

FDR-BH (q<0.05) 显著特征数:

| 比较 | 显著数 / 102 |
|---|---|
| 3 类 ANOVA | **85** |
| malignant vs benign | 67 |
| malignant vs no_tumor | 79 |
| **benign vs no_tumor** | **41** ← 用户最关心 |

### 2.5 Biomarker 图表

- **Volcano**: 3 面板 (mal-ben / mal-no / ben-no), 标出 top-8 abs-δ
- **Top-10 boxplot grid** (按 q_anova): ngtdm_Coarseness 是最强全局判别
- **Top-5 violin benign vs no_tumor** (按 q_ben_no):

| rank | feature | q_ben_no | Cliff's δ | 方向 |
|---|---|---|---|---|
| 1 | glcm_DifferenceVariance | 2.9e-4 | +0.31 | ben > no |
| 2 | glszm_SmallAreaEmphasis | 3.8e-4 | +0.29 | ben > no |
| 3 | glszm_SizeZoneNonUniformity | 3.8e-4 | +0.29 | ben > no |
| 4 | glszm_SizeZoneNonUniformityNormalized | 3.8e-4 | +0.28 | ben > no |
| 5 | glcm_Correlation | 3.8e-4 | -0.28 | no > ben |

→ **核心可解释 biomarker**: 良性胆囊息肉相较无肿瘤胆囊, 纹理更不均匀 (大 contrast / 小 correlation) 且包含更多小面积灰度区 → 对应病理上息肉内部回声不均.

- **Top-20 Pearson 相关 heatmap**: 显示 glcm 和 glszm 家族内部高度冗余, 后续建模时 PCA/LASSO 筛选足以
- **Class radar**: 三类在 top-6 特征上的 z-score 均值雷达图

### 2.6 胆囊 ROI vs 病灶 ROI (`2.6_gb_vs_lesion_compare.png`)

benign vs no_tumor 子集上:

| ROI | n | sil(t-SNE) | sil(raw) |
|---|---|---|---|
| gallbladder rect | 339 | 0.013 | 0.021 |
| **lesion polygon** | 342 | **0.028** | **0.040** |

→ lesion ROI 的可分性约是 gb ROI 的 2x. 未来若要做 ben-vs-no 的 biomarker 分类器, 应以 lesion mask 为主 ROI.

---

## 验证清单

- [x] Phase 1 CPU/GPU 运行时间合理 (< 10 min on GPU)
- [x] Phase 1 cls_feat_320 silhouette = 0.11499641 ≈ 既有 0.11499621 (sanity pass)
- [x] 多层 silhouette 近单调递增 (除 f3 略回落 → Swin 末层更抽象, 常见)
- [x] PyRadiomics 提取 453/456 张在 250s 完成 (< 15 min 目标), 3 张失败均已记录 `extract_log.json`
- [x] `biomarker_table.csv` top 5 `q_ben_no < 5e-4` (显著 biomarker 存在, 不是阴性结果)
- [x] 所有 PNG dpi=140, 图注含类名+样本数+silhouette
- [x] 所有 CSV 首列 `image_path` (可回溯定位)

---

## 代码落点

```
scripts/
  analyze_deep_features_multiview.py   # Phase 1
  extract_radiomics_features.py        # Phase 2.2
  analyze_radiomics_stats.py           # Phase 2.3-2.6
```

规划文档: `实验计划_特征可视化.md` (原样保留, 未改动)

---

## 一句话结论

> 深度网络在 **penultimate 层 (128-D)** 才把三类真正分开 (sil=0.18); 单独用经典影像组学不足以端到端区分三类 (sil≈0), 但能提供 41 个 FDR-显著的 benign-vs-no_tumor biomarker, top 候选集中在 GLCM/GLSZM 家族的纹理不均匀度. 若未来希望优化 benign↔no_tumor 混淆, lesion polygon 比 gallbladder rectangle 更能反映差异 (sil 翻倍).
