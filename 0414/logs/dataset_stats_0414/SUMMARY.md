# 0414dataset 统计学特征分析汇总

## 1) 数据规模
- 恶性 (malignant): 386 张
- 良性 (benign): 438 张
- 无肿瘤 (no_tumor): 1400 张
- 非矩形 polygon 总数: 2117
- 含 polygon 图像数: 1837 / 2224

## 2) 良性 vs 无肿瘤：polygon 最小外接矩形/形状差异
- 显著特征数 (q<0.05): 11 / 15
- poly_area_ratio_gb: q=2.90e-20, δ=0.29, 良性 > 无肿瘤
- mbr_area_ratio_gb: q=4.62e-20, δ=0.28, 良性 > 无肿瘤
- poly_area: q=4.57e-18, δ=0.26, 良性 > 无肿瘤
- poly_area_ratio_img: q=4.57e-18, δ=0.26, 良性 > 无肿瘤
- mbr_area: q=8.08e-18, δ=0.26, 良性 > 无肿瘤
- mbr_area_ratio_img: q=8.08e-18, δ=0.26, 良性 > 无肿瘤
- poly_perimeter: q=2.51e-17, δ=0.26, 良性 > 无肿瘤
- mbr_h: q=9.57e-15, δ=0.23, 良性 > 无肿瘤

## 3) 良性 vs 无肿瘤：图像级统计差异
- 显著特征数 (q<0.05): 9 / 17
- poly_area_mean: q=2.41e-16, δ=0.27, 良性 > 无肿瘤
- poly_mbr_area_mean: q=4.11e-16, δ=0.27, 良性 > 无肿瘤
- poly_area_max: q=1.24e-15, δ=0.26, 良性 > 无肿瘤
- poly_area_total_ratio_gb: q=2.60e-15, δ=0.26, 良性 > 无肿瘤
- poly_area_total: q=5.92e-13, δ=0.23, 良性 > 无肿瘤
- img_mean: q=3.73e-04, δ=-0.12, 无肿瘤 > 良性
- gb_mean: q=9.79e-04, δ=-0.11, 无肿瘤 > 良性
- poly_count: q=1.45e-03, δ=-0.06, 无肿瘤 > 良性

## 4) 三分类：共有图像特征差异（Kruskal）
- 显著特征数 (q<0.05): 14 / 19
- img_p10: q=6.61e-11, 均值[恶/良/无]=[22.33, 25.75, 28.79]
- img_p50: q=1.03e-08, 均值[恶/良/无]=[67.03, 70.17, 74.79]
- gb_p50: q=8.56e-08, 均值[恶/良/无]=[66.78, 69.45, 73.53]
- img_mean: q=7.60e-07, 均值[恶/良/无]=[73.93, 75.15, 79.72]
- img_skew: q=1.19e-06, 均值[恶/良/无]=[0.74, 0.62, 0.60]
- gb_mean: q=5.54e-06, 均值[恶/良/无]=[74.35, 75.22, 79.11]
- gb_p10: q=7.86e-05, 均值[恶/良/无]=[24.46, 24.55, 27.49]
- img_std: q=3.21e-04, 均值[恶/良/无]=[44.04, 40.53, 41.41]
- img_kurtosis: q=7.01e-03, 均值[恶/良/无]=[0.60, 0.32, 0.27]
- gb_rect_area: q=7.01e-03, 均值[恶/良/无]=[79119.83, 75566.91, 77513.64]

## 5) 英文术语解释（对照表）

### 5.1 类别与标注
- malignant: 恶性
- benign: 良性
- no_tumor: 无肿瘤
- polygon: 多边形分割标注（非矩形）
- MBR (minimum bounding rectangle): 最小外接矩形（本分析里是可旋转的最小外接矩形）
- gb: gallbladder，胆囊（通常指胆囊矩形框 ROI）
- img: 整张图像

### 5.2 统计检验符号
- p-value: 原始显著性概率，越小越显著
- q-value: 多重比较校正后的显著性（FDR-BH），通常 q<0.05 认为显著
- δ (Cliff's delta): 效应量，表示两组分布差异方向和强度
- δ > 0: 前一组更大（这里通常是“良性 > 无肿瘤”）
- δ < 0: 后一组更大（这里通常是“无肿瘤 > 良性”）
- Kruskal: Kruskal-Wallis 三组非参数检验

### 5.3 特征名前缀/后缀
- *_mean: 均值
- *_max: 最大值
- *_std: 标准差（离散程度）
- *_p10 / *_p50 / *_p90: 10% / 50% / 90% 分位数（p50=中位数）
- *_skew: 偏度（分布是否偏向一侧）
- *_kurtosis: 峰度（分布尖峭程度）
- *_entropy: 熵（灰度信息复杂度）
- *_count: 数量
- *_ratio_*: 比例（相对大小）

### 5.4 本文里常见变量逐个解释
- poly_count: 每张图里多边形病灶标注个数
- poly_area: 多边形面积（像素）
- poly_perimeter: 多边形周长（像素）
- poly_compactness: 形状紧致度，越接近圆通常值越大
- poly_area_mean / poly_area_max / poly_area_total: 单图多边形面积的平均/最大/总和
- poly_area_ratio_img: 多边形面积 / 整图面积
- poly_area_ratio_gb: 多边形面积 / 胆囊矩形面积
- poly_area_total_ratio_gb: 单图多边形总面积 / 胆囊矩形面积
- mbr_w / mbr_h: 最小外接矩形的宽/高
- mbr_area: 最小外接矩形面积
- mbr_aspect: 最小外接矩形长宽比（越大越细长）
- mbr_fill_ratio: 多边形面积 / 最小外接矩形面积（越大说明越“填满”外接框）
- mbr_area_ratio_img: 最小外接矩形面积 / 整图面积
- mbr_area_ratio_gb: 最小外接矩形面积 / 胆囊矩形面积
- mbr_center_x_norm / mbr_center_y_norm: 外接矩形中心点在图内归一化坐标（0~1）
- gb_rect_area: 胆囊矩形框面积
- gb_rect_area_ratio_img: 胆囊矩形框面积 / 整图面积
- img_mean / gb_mean: 整图 / 胆囊ROI 的平均灰度（亮度）

### 5.5 一眼看懂当前结果
- 良性 vs 无肿瘤：病灶相关面积特征（poly_area、mbr_area 及各种 ratio）在良性显著更大。
- 无肿瘤 vs 良性：整体灰度均值（img_mean、gb_mean）更高一些。
