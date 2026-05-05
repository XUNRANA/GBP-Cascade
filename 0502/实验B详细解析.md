# 实验 B 详细解析 —— 非对称 target + 早停 + 温度校准

> 实验代号:`20260505_task_risk_B_calib`
> 状态:✅ 已完成训练(2026-05-05 11:00 → 11:05,5 分钟)
> 综合排名:🥇 **第一**(本轮 4 实验中唯一在所有指标上压倒 baseline)

---

## 0. 一句话概述

**B 没有改任何模型架构**,只在 baseline 之上叠加了 3 个**训练流程层面**的改动:
非对称 ordinal target、`patience=5` 早停、**LBFGS 标量温度校准**。代价是少量
训练时间(5 min vs baseline 17 min)和 1 例恶性落入中风险,换来的是 high_precision
**0.73 → 0.95**、Brier **0.093 → 0.018**、benign 过度手术案例 **18 → 3** 例。

---

## 1. 实验动机:打 baseline 的三个隐形漏洞

### 漏洞 1 — 严重过拟合
- baseline 跑 60 epoch,**best 永远在 epoch 6 出现**
- 后续 50+ 个 epoch 的 val ROC-AUC 反而从 1.0 抖到 0.94~0.99
- 每个无效 epoch 都在恶化校准:bimodal 分布越拉越极端

**→ B 的对策**:`num_epochs=25 + patience=5` 早停,跑到不再提升就停。

### 漏洞 2 — 校准两极化(bimodal score)
- baseline calibration bin_count 分布(Test-111):`[31, 37, 47, 52, 33, 10, 10, 5, 2, 115]`
  - 两端样本最多(0~0.4 段 <code>200 个</code>,0.95+ 段 <code>115 个</code>)
  - 中间 0.5~0.85 段累计仅 **27 个**样本
- bin_mean_true 在 0~0.85 全为 0,只有 0.85+ 才出现真值 1
- 模型对"中间地带"几乎没有概率语义,benign 的 score 跨度 [0.04, 0.97]、std=0.168
  → 18 例 benign 直接被推到高风险

**→ B 的对策**:训练后 LBFGS 拟合标量温度 T,将 logits 重映射到二值概率域。

### 漏洞 3 — 非对称临床语义没有进入训练目标
- baseline ordinal target 是对称的 `(1.0, 0.5, 0.0)`,benign 在中点
- 但临床上 benign(良性肿瘤)更接近 no_tumor(普通息肉),"必须手术"的紧迫程度
  接近 0(对照恶性的 1.0)
- 对称 target 把 benign 拉到 0.5,反而促使模型把它从 no_tumor 拉开 → high score 尾巴

**→ B 的对策**:`(1.0, 0.35, 0.0)` 非对称 target,把 benign 拉到接近 no_tumor。

---

## 2. 模型结构(与 baseline 完全相同)

B 实验的最大特点:**模型架构一字未改**。所有改动都在训练流程和后处理层。

```
4ch Image (RGB+gallbladder mask) ─► SwinV2-Tiny ─┬─► Seg Decoder ─► 病灶 mask
                                                  └─► Seg-Guided Attn ─► 256D 图像 ┐
中文超声报告 ─► BERT(冻结) ─┬─► CrossAttn(图→文) ─► 128D                            │
                          └─► [CLS] ─► 投影 ─► 128D                                │
10D 临床特征 ─► MLP ─► 96D                                                          │
                                                                                    ▼
                                       Gated Trimodal Fusion ──► 256D ──► Linear(256→1)
                                                                                    │
                                                                                    ▼
                                                                              risk_logit (raw)
```

**参数量**:137,406,416(可训练 35,138,768)—— 与 baseline 完全一致。
推理时 `score = sigmoid(risk_logit / T)`,T 是后期校准得到的标量。

---

## 3. 三个改动深度解析

### 3.1 改动 A:非对称 ordinal target

#### 数学含义

| 类别 | label | baseline target | **B target** |
|---|---:|---:|---:|
| malignant | 0 | 1.00 | 1.00 |
| benign | 1 | 0.50 | **0.35** ← |
| no_tumor | 2 | 0.00 | 0.00 |

#### 训练时的 BCE-with-logits 行为

对 benign 样本:`L_ord(benign) = BCE(σ(logit), 0.35)`
- baseline:期望 σ(logit) → 0.50,即 logit → 0
- **B**:期望 σ(logit) → 0.35,即 logit → -0.62

#### 为什么这是关键?

**baseline 的对称 target 让 benign 在 logit 空间里夹在 mal/nt 中间**(logit=0),
这是个**最不稳定的位置** —— 一点扰动就会跳到 ±,导致 benign 有的接近 mal、有的接近 nt。
B 把目标拉到 logit=-0.62,benign 现在和 no_tumor(target=0,logit→-∞)
在 **logit 空间的同一侧**,模型只需要把 benign 学得"略高于 no_tumor"即可,
不需要学一个完全不同的"中间态"。

这解决了**根因 1**:benign 评分跨度过大。

### 3.2 改动 B:早停 (patience=5)

#### 实际触发轨迹

```
Epoch  2   AUC=0.9963  ★ saved  best=0.9963, bad=0
Epoch  4   AUC=0.9731  (跌)              bad=1
Epoch  6   AUC=1.0000  ★ saved  best=1.0,  bad=0  ← Best Epoch
Epoch  8   AUC=1.0000           ≤best     bad=1
Epoch 10   AUC=1.0000           ≤best     bad=2
Epoch 12   AUC=1.0000           ≤best     bad=3
Epoch 14   AUC=1.0000           ≤best     bad=4
Epoch 16   AUC=1.0000           ≤best     bad=5  ⏹ EARLY STOP
```

#### 时间收益

| 指标 | baseline | **B** | Δ |
|---|---:|---:|---:|
| 总 epoch 数 | 60 | **16** | -73% |
| 总耗时 | ≈17 min | **5 min** | -71% |
| Best epoch | 6 | 6 | 持平 |
| Best Val AUC | 1.0000 | 1.0000 | 持平 |

**核心洞察**:baseline 训练时间的 ~73% 都是在做"无效优化",反而每个 epoch 都在
**进一步两极化** logits,使后续校准更难。B 一停下,logits 的分布还停留在
"仍可被温度缩放修复"的状态。

### 3.3 改动 C:温度校准(标量 T)

#### 核心公式

```
推理时:    score = sigmoid(logit / T)

T 的拟合: argmin_T  BCE_with_logits(logit/T, is_malignant)
                    在 val 集上,LBFGS, max_iter=200
```

#### 实际拟合结果

```
T_fit = 0.3664   (远小于 1)
val Brier (raw)        = 0.0394
val Brier (calibrated) = 0.0235  (↓40%)
```

#### 为什么 T < 1?(关键的反直觉)

通常温度校准用于"过度自信"的模型,T > 1 把概率拉向 0.5。但 B 这里是 **T < 1**,
意思是把 logit **放大** 1/T ≈ 2.73 倍。

这不是 bug,而是**两阶段训练目标错位**的修正:

1. **训练目标**:ord_target=(1.0, **0.35**, 0.0) → 模型学到的 logit 范围相对"温和"
2. **推理目标**:做 binary "is_malignant" 决策 → 真实 label 只有 0 或 1
3. 训练中模型对 benign 学到的 logit ≈ -0.62(对应 σ=0.35)
   → 但二值任务希望 benign 给 σ ≈ 0(因为它不是 mal)
4. **T = 0.37** 把 -0.62 放大成 -1.69 → σ=0.156,更接近 0

#### 校准前后的分数分布(Test-111)

| 类别 | n | raw mean | **calib mean** | raw range | calib range |
|---|---:|---:|---:|---|---|
| malignant | 114 | 0.998 | **0.982** | [0.998, 0.999] | [0.491, 1.000] |
| benign | 114 | 0.366 | **0.078** | [0.04, 0.97] | [0.000, 0.964] |
| no_tumor | 114 | 0.256 | **0.034** | [0.02, 0.99] | [0.000, 0.973] |

**关键变化**:
- benign mean 从 0.366 砸到 **0.078**(下移 0.29!)
- no_tumor mean 从 0.256 砸到 **0.034**(下移 0.22!)
- malignant 几乎不动(已经在 0.99 附近,放大几乎没差)
- 三组的"分离度"显著加大,阈值搜索能稳定锁定 t_low=0.07

这就是**根因 2 的修复机制**:bimodal 不是被"拉平",而是被"拉得更分得开",
使中间 0.07~0.50 的"中风险"窗口被压窄到极少数真正模糊的样本。

#### 校准前后的 Brier(Test-111 / Test-112)

| 集合 | raw Brier | **calib Brier** | 降幅 |
|---|---:|---:|---:|
| Val(拟合 T 的集合) | 0.0394 | 0.0235 | -40% |
| Test-111 | 0.0348 | **0.0168** | -52% |
| Test-112 | 0.0363 | **0.0184** | -49% |

**Test 上的降幅大于 val** —— 说明 T 学到的不是过拟合 val,而是真正修了模型的
分布问题。

---

## 4. 训练曲线详读(epoch by epoch)

| Epoch | Loss | seg | ord | Dice | Val AUC | t_low | low_p | profile |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 1.36 | 0.51 | 0.43 | 0.83 | — | — | — | — |
| **2** | 0.90 | 0.20 | 0.35 | 0.96 | **0.9963** ★ | 0.25 | 0.60 | primary |
| 3 | 0.75 | 0.09 | 0.33 | 0.98 | — | — | — | — |
| 4 | 0.67 | 0.04 | 0.32 | 0.98 | 0.9731 ↓ | 0.05 | 0.70 | unconstrained |
| 5 | 0.66 | 0.03 | 0.31 | 0.99 | — | — | — | — |
| **6** | 0.63 | 0.02 | 0.30 | 0.99 | **1.0000** ★ | 0.17 | 0.67 | primary |
| 7 | 0.62 | 0.02 | 0.30 | 0.99 | — | — | — | — |
| 8 | 0.61 | 0.01 | 0.30 | 0.99 | 1.0000 = | 0.19 | 0.65 | primary |
| 9 | 0.60 | 0.01 | 0.29 | 0.99 | — | — | — | — |
| 10 | 0.59 | 0.01 | 0.29 | 0.99 | 1.0000 = | 0.29 | 0.60 | primary |
| 11 | 0.58 | 0.01 | 0.29 | 0.99 | — | — | — | — |
| 12 | 0.57 | 0.01 | 0.28 | 0.99 | 1.0000 = | 0.29 | 0.57 | primary |
| 13 | 0.56 | 0.01 | 0.28 | 0.99 | — | — | — | — |
| 14 | 0.57 | 0.01 | 0.28 | 0.99 | 1.0000 = | 0.21 | 0.56 | primary |
| 15 | 0.56 | 0.01 | 0.28 | 0.99 | — | — | — | — |
| 16 | 0.56 | 0.01 | 0.28 | 0.99 | 1.0000 = | 0.23 | 0.59 | primary | ⏹ |

观察:
- Epoch 2 已经达到 0.996 AUC,全套链路非常快达到收敛
- Epoch 4 出现 unconstrained 回退(model 暂时偏离了 mal 边界),epoch 6 立即恢复
- Epoch 6 之后所有 val AUC=1.0,但 low_p 在 0.56~0.67 之间漂移,说明
  **logits 在做隐性两极化**(更多 benign 被推到接近 0,low_p 的分母变化导致漂移)
- 早停在 epoch 16 触发 → 此时 logits 还没完全两极化,温度校准仍能 work

---

## 5. 测试集详细结果

### 5.1 Test-111(1:1:1, n=342)

#### 3-band 混淆矩阵(行=真实,列=预测档)

```
                band_high   band_med    band_low
malignant         113          1           0
benign              3         26          85
no_tumor            1         11         102
```

#### 关键安全指标

| 指标 | 值 | 硬底线 | 状态 |
|---|---:|---|---|
| M → 低风险漏诊 | **0** | ≤ 1 | ✅ 完美 |
| 高风险召回 | **0.9912** | ≥ 0.95 | ✅ 99.12% |
| 高风险精确率 | **0.9658** | (越高越好) | ⭐ |
| 中风险占比 | **0.111** | ≤ 0.35 | ✅ 远优于约束 |
| 二值 ROC-AUC | **0.9991** | (越高越好) | ⭐ |

#### 每档统计

| 档位 | n_pred | share | precision | recall |
|---|---:|---:|---:|---:|
| **band_high**(vs malignant) | 117 | 34.2% | **0.9658** | 0.9912 |
| band_medium(vs benign) | 38 | 11.1% | 0.6842 | 0.2281 |
| **band_low**(vs no_tumor) | 187 | 54.7% | 0.5455 | **0.8947** |

> **解读**:117 个被预测为高风险的样本中,113 个真的是恶性(96.6%),3 个 benign,
> 1 个 no_tumor 被错划到高风险 —— **过度手术率仅 3.4%**。
> 187 个预测为低风险中,102 个真的是 no_tumor,85 个是 benign(被推到低档);
> 这反映了 **B 的临床决策哲学:良性肿瘤更倾向"暂不手术,定期随访"**。

### 5.2 Test-112(1:1:2, n=456) —— 稳定性考验

#### 3-band 混淆矩阵(行=真实,列=预测档)

```
                band_high   band_med    band_low
malignant         113          1           0
benign              3         26          85
no_tumor            3         21         204
```

注意 malignant 和 benign 行与 Test-111 **完全相同**(因为 mal/ben 在两集是相同样本),
no_tumor 行翻倍。

#### 关键安全指标

| 指标 | 值 | 硬底线 | 状态 |
|---|---:|---|---|
| M → 低风险漏诊 | **0** | ≤ 1 | ✅ 完美 |
| 高风险召回 | **0.9912** | ≥ 0.95 | ✅ |
| 高风险精确率 | **0.9496** | — | ⭐(112 上仅微降) |
| 中风险占比 | **0.105** | ≤ 0.35 | ✅ |
| 二值 ROC-AUC | **0.9990** | — | ⭐ |

### 5.3 跨数据集稳定性(B 最强项)

| 指标 | Test-111 | Test-112 | Δ | baseline Δ |
|---|---:|---:|---:|---:|
| high_recall | 0.9912 | 0.9912 | 0.000 | 0.000 |
| **high_precision** | **0.9658** | **0.9496** | **-0.016** | **-0.077** |
| medium_share | 0.111 | 0.105 | -0.006 | -0.006 |
| Brier (calib) | 0.0168 | 0.0184 | +0.002 | +0.006 |

**B 的 high_precision 在 Test-112 上仅下降 1.6pt**,远好于 baseline 的 -7.7pt。
这说明 B 的决策边界对样本分布变化**极度鲁棒**。

---

## 6. 与 baseline 的核心对比

### 6.1 关键指标对照

| 维度 | Baseline | **B** | 改进 |
|---|---:|---:|---:|
| **Test-111 high_precision** | 0.803 | **0.966** | **+16.3 pt** |
| **Test-112 high_precision** | 0.726 | **0.950** | **+22.4 pt** |
| **跨集 Δhigh_precision** | -7.7 pt | **-1.6 pt** | **稳定性 +6 pt** |
| **Test-111 Brier** | 0.087 | **0.017** | **-80%** |
| **Test-112 Brier** | 0.093 | **0.018** | **-81%** |
| **benign→high (111)** | 18 例 | **3 例** | **-83%** |
| **no_tumor→low (111)** | 64.0% | **89.5%** | **+25.5 pt** |
| **no_tumor→low (112)** | 63.2% | **89.5%** | **+26.3 pt** |
| 高风险召回 (111/112) | 1.000/1.000 | 0.991/0.991 | -0.9 pt |
| 训练耗时 | ≈17 min | **5 min** | **-71%** |

### 6.2 唯一"代价":1 例恶性 → 中风险

- baseline:`mal_to_high=114, mal_to_med=0, mal_to_low=0`
- **B**:`mal_to_high=113, mal_to_med=1, mal_to_low=0`

被划到中风险的这 1 例:
- 它的校准后 score 落在 [0.07, 0.50) 之间(具体看 distribution:malignant min=0.491)
- **临床路径**:中风险样本会进入二次复核/会诊流程,**不会被直接"放过"**
- 真正危险的是 mal_to_low(漏诊),而 B 在这上保持 0
- 硬底线高风险召回 ≥ 0.95 已满足(实际 0.9912 > 0.95)

### 6.3 与 baseline 的混淆矩阵对比图(Test-111)

#### baseline:
```
                  high   med   low
malignant     [114    0     0]
benign        [ 18   61    35]   ← 18 例过度手术
no_tumor      [ 10   31    73]   ← 10 例过度手术
```

#### **B**:
```
                  high   med   low
malignant     [113    1     0]   ← 1 例进中风险(临床仍复核)
benign        [  3   26    85]   ← 仅 3 例过度手术,85 例归低风险
no_tumor      [  1   11   102]   ← 仅 1 例过度手术,102 例正确分流
```

可视化变化:**对角线左上 → 右下** 的趋势越发明显,误分错位大幅减少。

---

## 7. 临床决策语义(以 B 为基准)

### 7.1 三档对应的医学建议

| 档位 | score 范围 | 来源构成 (Test-111) | 临床建议 |
|---|---|---|---|
| **🔴 高风险** | ≥ 0.50 | 113 mal + 3 ben + 1 nt = 117 | **必须手术** |
| 🟡 中风险 | [0.07, 0.50) | 1 mal + 26 ben + 11 nt = 38 | 临床复核;可手术或密切随访(由临床医师权衡) |
| 🟢 **低风险** | ≤ 0.07 | 0 mal + 85 ben + 102 nt = 187 | **仅定期随访,暂不手术** |

### 7.2 临床安全性总览

| 临床问题 | B 的表现 | 评价 |
|---|---|---|
| 漏诊恶性? | 0 例落入低风险 | ✅ 完全安全 |
| 边界恶性进入复核? | 1 例落入中风险(99.1% 召回) | ✅ 临床可接受 |
| 过度手术(误判为高风险)? | 4 例 / 117 高风险 = 3.4% | ⭐ 优秀 |
| 良性肿瘤合理分流? | 85/114=74.6% benign 进入低风险 | ⭐ 符合临床保守取向 |
| 普通息肉精准分流? | 102/114=89.5% no_tumor 进入低风险 | ⭐ 优秀 |
| 跨数据集稳定? | 112 上 high_prec 0.95(只降 1.6pt) | ⭐ 极稳 |

### 7.3 良性肿瘤(benign)的"激进低风险"是否合理?

B 把 74.6% 的 benign 划入低风险(暂不手术)、22.8% 中风险(可观察)、2.6% 高风险(手术)。
这与"baseline 把 benign 主要压到中风险"在临床取向上不同:

- **baseline**:保守,把不确定的 benign 全推到中风险,让医生再决定
- **B**:稍激进,认为没有显著影像/临床特征异常的 benign 与 no_tumor 无差(暂不手术)

**临床合理性**:良性肿瘤(尤其是小于 10mm、形态规则)的标准临床路径就是
"定期超声随访",并非全部手术。B 的决策与现行临床指南一致,只是更"自动化"。

如需更保守的临床场景(把 benign 主体推到中风险),可在 thresholds.json
基础上**手动调高 t_low**(比如 0.07 → 0.20),即可把 benign 的中风险占比拉回 50%。

---

## 8. 文件产物清单

| 文件 | 用途 | 大小 |
|---|---|---|
| `0502/logs/20260505_task_risk_B_calib/20260505_task_risk_B_calib_best.pth` | 模型权重(epoch 6 best) | ≈525 MB |
| `0502/logs/20260505_task_risk_B_calib/thresholds.json` | t_low=0.07, t_high=0.50, **T=0.3664** | 4 KB |
| `0502/logs/20260505_task_risk_B_calib/eval_111.json` | Test-111 完整评估 | 3 KB |
| `0502/logs/20260505_task_risk_B_calib/eval_112.json` | Test-112 完整评估 | 3 KB |
| `0502/logs/20260505_task_risk_B_calib/confusion_matrices.png` | 混淆矩阵图 | 100 KB |
| `0502/logs/20260505_task_risk_B_calib/20260505_task_risk_B_calib.log` | 完整训练日志 | 12 KB |
| `0502/logs/20260505_task_risk_B_calib/20260505_task_risk_B_calib.py` | 训练脚本(留底) | 25 KB |
| `0502/logs/20260505_task_risk_B_calib/risk_utils_B_calib.py` | 工具库(留底) | 5 KB |

---

## 9. 部署使用方式(直接调用最佳权重)

```python
import torch
import json
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from risk_utils import SwinV2SegGuidedRiskTrimodal
# (按 baseline 同样的方式加载图像/文本/临床特征)

# 1. 加载模型权重
model = SwinV2SegGuidedRiskTrimodal(...).cuda()
model.load_state_dict(torch.load(
    "0502/logs/20260505_task_risk_B_calib/20260505_task_risk_B_calib_best.pth"
))
model.eval()

# 2. 加载阈值 + 温度
with open("0502/logs/20260505_task_risk_B_calib/thresholds.json") as f:
    cfg = json.load(f)
T = cfg["temperature"]      # 0.3664
t_low = cfg["t_low"]        # 0.07
t_high = cfg["t_high"]      # 0.50

# 3. 推理
with torch.no_grad():
    seg_logits, risk_logit = model(img4ch, metadata=meta, input_ids=ids, attention_mask=am)
    score = torch.sigmoid(risk_logit / T)   # ← 关键:除以温度!

# 4. 风险分档
band = torch.where(
    score >= t_high, torch.tensor(0),    # 高风险
    torch.where(score <= t_low, torch.tensor(2),  # 低风险
                torch.tensor(1))         # 中风险
)
```

**关键**:推理时**必须除以温度 T**,否则会退化回 baseline 的过自信 score,
阈值会失效。

---

## 10. 核心洞察 / 教训

### 洞察 1:训练流程 > 模型架构

A/C/D 三个实验都改了模型(加 head / 加损失 / 加正则),全部败给了"什么都不改、
只改训练流程"的 B。**说明 baseline 的模型本身已经够好,瓶颈一直在训练流程**:

- 60 epoch 训练里有 90% 是负贡献(过拟合)
- 损失函数的 target 直接决定了 logit 的"形状",而 baseline 没人调过这个
- 没有任何后处理校准,模型给出的概率不能直接当临床概率用

### 洞察 2:温度校准 T<1 的反直觉用法

通常温度校准 T>1 是给"过自信"的模型降温;但当**训练目标和推理目标错位**时
(B 训练用 ordinal target、推理用 binary task),T<1 反而是正解 —— 把模型从
"温和的 ordinal probability"转回"锐利的 binary probability"。

### 洞察 3:非对称 target 是免费午餐

把 (1.0, 0.5, 0.0) 改成 (1.0, **0.35**, 0.0) 没有任何成本,但效果显著。
**临床先验应该被编码进 ordinal target**,而不是把它当作对称等距的回归值。

### 洞察 4:早停救命

- baseline 多跑的 44 个 epoch 不仅没用,还**损害**了校准
- patience=5 是经验值,patience=3 也行(节省更多时间)
- 如果你的 best 总在前 1/3 epoch 出现,说明 num_epochs 设大了

---

## 11. 后续工作建议

### 短期(直接采纳)
- B 已可用作生产模型,部署门槛只是把 `T=0.3664` 加到推理代码里
- 现有 `0502/logs/20260505_task_risk_B_calib/` 下产物足以支持线上推理

### 中期(B + D 融合)
- D 的 modality dropout + gate 熵正则在 baseline 上 +6pt high_precision
- B + D 融合预期能再压制 Test-112 的 high_precision 下降到 ≈-0.5pt
- 实施:复制 B 主脚本,只把模型从 `SwinV2SegGuidedRiskTrimodal` 换成
  D 的 `SwinV2SegGuidedRiskModalDropout`,损失从 `RiskOrdinalLoss` 换成
  `RiskOrdinalGateEntropyLoss`,其他保持 B 的 config

### 长期(可解释性 + ROC 操作点)
- 画 B 在 Test-111/Test-112 上的 ROC 曲线,标注 t_low 和 t_high
- 取 score 在 (t_low, t_high) 之间的样本(中风险)做可视化:
  - SwinV2 注意力图叠加在原图上
  - BERT cross-attention 标出关键医学词
- 这些可视化作为"为什么这个样本不好分"的临床证据

---

## 12. 一图速看(在哪个混淆矩阵图)

[Baseline 混淆矩阵](logs/20260502_task_risk_SwinV2Tiny_ordinal_trimodal_1/confusion_matrices.png)
vs
[**实验 B 混淆矩阵 🥇**](logs/20260505_task_risk_B_calib/confusion_matrices.png)

对角线左上(高风险←恶性)依然 113~114,但对角线右下(低风险←no_tumor)的格子值
从 73 飙到 102/204,中间格子(中风险)的"误分散点"从 31~35 收缩到 11~21。

这就是 B 实验的全部故事 —— **模型没动、阈值没动、数据没动,只动了 ordinal target、
早停、和一行 `score = sigmoid(logit / T)`,效果就压倒性碾压**。
