# Transformer 主线推进与 80% 目标方案

## 1. 先说结论

如果只看现有仓库里的全部实验，结论非常明确：

1. 现在最值得继续推进的不是再换一堆新 backbone，而是把 **Transformer 主线** 做深。
2. 这条主线里最有希望的不是纯 ROI，也不是纯全图，而是 **全图上下文 + ROI 细节 + JSON 标注辅助监督**。
3. 如果目标是把 `Acc / Precision / Recall / F1(macro)` 四个指标都拉到 `80%`，**单靠当前这批图像和当前训练流程，概率很低**。
4. 但这不等于没法继续推进。正确做法是：
   - 模型侧：把 Transformer 主线做到当前数据条件下的上限
   - 流程侧：把验证/阈值/多 seed 做规范
   - 数据侧：针对 80% 目标补强真正的短板

这份文档分两件事讲：

- 之前所有尝试到底证明了什么
- 现在应该怎么推进一个真正有逻辑的 Transformer 模型

---

## 2. 为什么必须先统一视角

仓库里目前有两批核心实验：

- `0319/`：基于 `0316dataset/Processed`
- `0323/`：基于 `0322dataset`

它们不是同一套 train/test，也不是同一个阶段的问题。

### 2.1 `0319` 主要解决的是“把模型训对”

这批实验的重点是：

- baseline 到底长什么样
- class imbalance 怎么处理
- 哪种 optimizer / scheduler / augment recipe 更适合不同 backbone
- 谁能成为当前主线模型

### 2.2 `0323` 主要解决的是“怎么把 JSON 标注吃进去”

这批实验的重点是：

- ROI 裁剪有没有用
- lesion mask 第 4 通道有没有用
- 全图上下文是不是比 ROI 更重要
- 强增强、Mixup、Focal、阈值调优有没有额外收益
- 集成、stacking、特征工程还能不能继续突破

所以正确的阅读方式不是把两批实验混成一张排行榜，而是：

- `0319` 看“recipe 和 backbone”
- `0323` 看“标注利用和 Transformer 主线”

---

## 3. 之前所有尝试，按问题来重排

下面不按时间线，而按“试图解决什么问题”来重排。

---

## 4. 尝试一：基础 CNN / Transformer baseline

### 4.1 试图解决的问题

最初想回答的是：

> 这个任务上，换 backbone 能不能直接带来明显提升？

### 4.2 实际做了什么

第一批 baseline 大致包括：

- ResNet18
- ResNet34
- DenseNet121
- EfficientNet-B0
- ConvNeXt-Tiny
- Swin-Tiny
- ViT-B16

大多采用统一 recipe：

- ImageNet 预训练
- 224 输入
- Adam
- 固定学习率
- 全量微调

### 4.3 结果告诉了我们什么

最重要的结论不是“谁最高”，而是：

1. 单看 `Acc` 完全不够。
2. 某些 Transformer baseline 会塌成“几乎全预测 `no_tumor`”。
3. 早期很多差结果，不一定是 backbone 不行，而是训练 recipe 不对。

比如：

- `SwinTiny_baseline` 虽然 `Acc` 很高，但 `benign recall = 0`
- 这类结果说明模型并没有真正学会任务，只是学会了多数类偏置

### 4.4 这一步的结论

> baseline 阶段没有回答“谁最强”，只回答了“这个任务极度不适合只看 accuracy，而且统一 recipe 会把一部分 Transformer 直接训废”。

---

## 5. 尝试二：class weight、weak augmentation、sampler

### 5.1 试图解决的问题

> 当前任务最大问题是不是类别不平衡？

答案是：是，而且影响很大。

### 5.2 做了哪些方向

- `class weight`
- `weak augmentation`
- `WeightedRandomSampler`
- `1:1` train/test 的平衡实验

### 5.3 学到了什么

#### 结论 A：`class weight` 是必要项

这一点几乎在两批实验中都反复被验证：

- 不加权时，模型天然偏向 `no_tumor`
- 加权后，`benign recall` 才有机会起来

#### 结论 B：augmentation 不能单独解决问题

一个非常典型的发现是：

- `weakaug` 单独上会变差
- `weakaug + classweight` 会明显变好

这说明：

- 增强只是增加训练难度
- 但不平衡处理才是决定模型是否愿意学习 `benign` 的关键

#### 结论 C：sampler 不是万能药

`WeightedRandomSampler` 的作用更像：

- 提高 batch 内 `benign` 暴露频率
- 让模型不那么快倒向多数类

但从已有结果看，它并没有自动带来“全面更好”。

在 `0319` 的 `ConvNeXt img320` 对照里，sampler 更像：

- `precision/Acc` 更高
- 但 `benign recall` 不一定更强

在 `0323` 的 SwinV2 线上，sampler 只有和其他因素一起才有意义：

- sampler + 修复 Mixup 权重 + 阈值调优

### 5.4 这一步的结论

> `class weight` 是基础设施，augmentation 是辅助项，sampler 只是边界调节器，三者不能互相替代。

---

## 6. 尝试三：recipe-tuning

### 6.1 试图解决的问题

> 首轮那些“表现差的模型”到底是模型本身不行，还是训练 recipe 不匹配？

### 6.2 做了哪些改动

核心 recipe 包括：

- `AdamW`
- backbone / head 差分学习率
- `warmup + cosine`
- `class weight`
- `label smoothing`
- 更贴模型的 augmentation
- `grad clip`

### 6.3 结果非常明确

这一轮的真正含义是：

- `ConvNeXt`
- `Swin`
- `ViT`
- `EfficientNet`

都被明显“拉回来了”。

这说明：

- baseline 里很多模型不是“没能力”
- 而是“没被正确训练”

### 6.4 这一步对 Transformer 主线的意义

这一步非常关键，因为它为后面的判断打下了基础：

> 如果一个 Transformer 在合理 recipe 下仍然不行，那才是真的不行；如果是错误 recipe 下不行，那不能据此否掉 Transformer 路线。

---

## 7. 尝试四：ROI 裁剪、4 通道 mask、双分支

这一块主要发生在 `0323`。

### 7.1 ROI 裁剪：最初的直觉是合理的

最开始的想法是：

- 医生已经给了胆囊矩形框
- 先裁掉无关背景
- 让模型聚焦胆囊

这就是 `ROI 3ch` 和 `ROI 4ch` 的出发点。

### 7.2 ROI 4ch 比 ROI 3ch 更有价值

`Exp #3: SwinV2 ROI 4ch` 的意义非常大：

- 它证明了 lesion mask 第 4 通道是真的有用
- 模型不是只靠图像纹理，也在利用病灶位置先验

这一步说明：

> JSON 标注不是装饰，而是可以转化成有效监督和有效输入。

### 7.3 但 ROI 不是最终最优表示

这点是 `0323` 最值得重视的结论之一。

原本直觉上会认为：

- ROI 更干净
- 背景更少
- 应该更强

但实验结果最后反而指向：

- **全图 + mask** 比 **ROI + mask** 更有效

这意味着：

- ROI 裁剪在去噪的同时，也丢掉了上下文
- 对一个本来类间差异就弱的任务，这部分上下文可能比想象中更重要

### 7.4 双分支为什么没有跑赢

已有双分支实验是：

- 全局 ROI 分支
- 病灶局部分支
- 结构化特征分支

它没成为主线，原因不是思路错，而是：

1. 输入还是 3 通道，没吃到 mask 的收益
2. 没有沿用后面那条更强的 `full4ch + balanced + threshold` 主线
3. 模型复杂度高，batch 小，过拟合更容易
4. 分支设计抓的是 `ROI + lesion crop`，而不是 `full context + ROI detail`

这点很重要，因为它告诉我们：

> 双分支这个方向没有被彻底证伪，真正被证伪的是“旧版的双分支设计”。

---

## 8. 尝试五：强增强、Mixup / CutMix、Focal Loss、阈值调优

### 8.1 强增强有用，但要配合正确输入表示

`Exp #7` 的强增强组合之所以重要，不是因为增强本身多花哨，而是因为它搭在：

- `full image`
- `4ch mask`
- `SwinV2`

这条线上。

这说明强增强的收益依赖于：

- 正确的输入表示
- 正确的 backbone

### 8.2 Mixup 最开始其实被代码 bug 限制了

这是 `0323` 最关键的技术发现之一。

之前 Mixup 的 soft CE 没有乘类别权重，导致：

- 少数类在混合训练里失去额外权重

修复后，整条线的逻辑才真正成立：

- 既做数据混合正则化
- 又继续对 `benign` 敏感

### 8.3 Focal Loss 没有成为最优主线

这不代表 Focal 完全没价值，而是它在这个任务里表现为：

- 能短时间把难样本抓得更积极
- 但也更容易把边界推过头

从结果上看，它更像补充工具，而不是新主线。

### 8.4 阈值调优是必须正视的一个变量

`0323` 的结果里，很多模型在默认阈值下并不是最优。

这说明：

- 模型学到的并不只是“对 / 错”
- 还包括一种有偏的概率校准

因此，后续如果继续推 Transformer 主线，阈值调优必须作为正式组件处理，而不是赛后补丁。

---

## 9. 尝试六：multi-seed、ensemble、stacking

### 9.1 multi-seed 证明了什么

它证明了两件事：

1. 最好单模型成绩有随机性
2. 但这条线不是纯碰运气，因为多 seed 均值仍然在同一个水平区间

所以：

- 最好 seed 看潜力
- 多 seed 均值看稳定性

### 9.2 简单平均不够强

6 模型整体平均并没有明显超过最好单模型。

这说明：

- 模型越多不一定越好
- 关键是互补性，而不是数量

### 9.3 stacking 为什么有效

stacking 能到全仓当前最高的一档，是因为它在做一件比“简单平均”更聪明的事：

- 学习不同预测器在不同区域的可信度

这很适合现在这个阶段，因为当前瓶颈已经不是“模型完全不会”，而是：

- 模型都学到了一部分
- 但各自都有偏差

---

## 10. 尝试七：XGBoost、DINOv2、标注类型感知特征

### 10.1 深度特征 + XGBoost 的意义

这条线证明了：

- 神经网络输出不是唯一能用的信息
- 但树模型没有从已有特征里挖出一个全新的巨大增益

换句话说：

- 分类器头不是主要瓶颈

### 10.2 DINOv2 的意义

这条线说明：

- 通用自监督特征并不天然比任务微调特征更强

这对后续 Transformer 路线很重要，因为它提示我们：

- 继续堆“更大更通用”的 Transformer，不一定比“更贴任务的 Transformer 结构和监督信号”更有价值

### 10.3 标注类型感知的意义

它抓住了一个很重要的事实：

- `adenoma` 的确有鉴别力

但没变成大突破，原因是：

- 覆盖率低
- 对多数样本帮助有限

它更适合作为：

- Transformer 的辅助监督

而不是：

- 独立的主分类器

---

## 11. 之前所有尝试，真正留下了什么可用资产

如果把所有尝试都提纯，最后真正留下来的“可复用资产”只有这些：

1. `SwinV2` 是目前最值得继续深挖的 Transformer 骨干。
2. `4ch lesion mask` 是有效信息。
3. `full image` 上下文不能丢。
4. `ROI detail` 也不能完全丢。
5. `class weight + balanced sampler` 仍然是少数类基础设施。
6. `threshold tuning` 必须成为正式组件。
7. `annotation-aware` 信息更适合作为辅助监督，而不是替代主分类器。

这 7 条就是后面 Transformer 主线设计的依据。

---

## 12. 现在为什么必须推进一个 Transformer 模型

这不是因为“Transformer 听起来高级”，而是因为当前证据已经表明：

1. `Swin/SwinV2` 在少数类上更有侵略性。
2. `4ch mask` 与窗口注意力的组合效果是成立的。
3. Transformer 更容易做多视角融合和辅助任务头。
4. 继续在 CNN 线上堆，已经看不到足够清晰的新收益点了。

所以接下来继续做 Transformer，不是跟风，而是顺着现有证据往下推。

---

## 13. 下一条 Transformer 主线应该长什么样

### 13.1 不能再简单重复旧实验

不能重复的路线包括：

- 纯 ROI 4ch
- 纯 full4ch
- 旧版双分支 3ch
- 单纯把 Focal 再跑一遍
- 再随便换一个新 Transformer backbone

因为这些路径，要么已经跑过，要么性价比不高。

### 13.2 最合理的下一步：双视角 Transformer

最值得推进的结构是：

> **Full image 4ch + ROI 4ch dual-view SwinV2**

原因很直接：

- `Exp #7/#9` 证明了 full4ch 的上下文价值
- `Exp #3` 证明了 ROI4ch 的细节价值
- 这两条线的优势并没有真正被合并过

### 13.3 再加一层：annotation-aware auxiliary supervision

为什么要加辅助监督？

因为已有实验已经证明：

- 标注类型本身不足以单独分类
- 但它们不是没用

最合理的用法不是让它们直接接管主任务，而是：

- 作为 Transformer 表示学习的正则项

例如让模型同时预测：

- `has_polyp`
- `has_pred`
- `has_adenoma_any`
- `lesion_area_ratio`

这样做的目标不是直接靠辅助头涨分，而是强迫 backbone 学到：

- 更贴近病灶语义的表征

### 13.4 为什么这个结构比旧版双分支更靠谱

新主线相对旧版双分支，优势在于：

1. 不是 `ROI + lesion crop`，而是 `full context + ROI detail`
2. 两个分支都是 `4ch`
3. 继承了现有最强的 `SwinV2` 体系
4. 继承了 balanced sampler / class weight / threshold tuning
5. 辅助监督来自已知有价值但覆盖率不高的 JSON 语义

换句话说，这不是重复 Exp #6，而是把：

- `Exp #3`
- `Exp #7`
- `Exp #9`
- `annotation-aware`

四条有效信息合成一条新主线。

---

## 14. 我已经落下来的新实验脚本

新脚本路径：

- [20260324_task2_SwinV2Tiny_fullroi4ch_auxannot_12.py](/data1/ouyangxinglong/GBP-Cascade/0323/scripts/20260324_task2_SwinV2Tiny_fullroi4ch_auxannot_12.py)

它做的事情是：

1. 共享一个 `SwinV2-Tiny` 4 通道 backbone
2. 同时看 `full image 4ch` 和 `ROI 4ch`
3. 融合 `full / roi / abs diff / elementwise product`
4. 主头做 `benign vs no_tumor`
5. 辅助头预测：
   - `has_polyp`
   - `has_pred`
   - `has_adenoma_any`
   - `lesion_area_ratio`
6. 保留：
   - `balanced sampler`
   - `class weight`
   - `warmup + cosine`
   - `threshold tuning`

这是我认为当前最值得继续推进的 Transformer 主线版本。

---

## 15. 但要实话实说：单靠这个模型，80% 仍然大概率不够

### 15.1 为什么

因为现有所有实验已经反复给出同一个信号：

- 单模型大多在 `0.60 ~ 0.64`
- stacking 才把上限推到 `0.65` 左右

这不是一个“差一点就 80”的状态，而是一个“还差一大截”的状态。

### 15.2 80% 目标到底缺什么

如果真要把四个指标都做到 `80%`，缺口不是一个模型小改能补上的，而是至少包括：

1. **更干净的评估流程**
   - 现在 test 被反复用于模型选择和阈值选择

2. **更多高质量 benign 数据**
   - 少数类覆盖仍然不够

3. **标签质量和语义一致性核查**
   - 特别是 benign / no_tumor 的边界样本

4. **更强的辅助信息**
   - 临床信息
   - 病理信息
   - 更结构化的病灶描述

### 15.3 所以 80% 要分两层理解

#### 层 1：模型层面的 80% 冲刺

可以理解为：

- “在现有图像条件下尽量逼近更高上限”

这件事靠 Transformer 主线继续做是对的。

#### 层 2：业务目标层面的真实 80%

这件事则需要：

- 数据
- 标注
- 临床特征
- 规范验证

否则就算短期把某个 test 分数冲高，也很可能不可复现。

---

## 16. 如何把 Transformer 主线推进到当前上限

下面是我建议的实际推进顺序。

### 16.1 第一优先级：先跑新双视角 Transformer

目标：

- 验证“full + ROI + aux supervision”是否比已有单视角 SwinV2 更好

关键比较对象：

- `Exp #3` ROI4ch
- `Exp #7` full4ch strongaug
- `Exp #9` balanced mixup

希望看到的信号不是只看总 F1，而是：

- `benign recall` 有没有继续升
- `benign precision` 是否没有同步崩
- 调阈值后是否能稳定超过已有 Transformer 单模型

### 16.2 第二优先级：做 ablation，不要盲目叠改动

必须拆开验证：

1. full + roi 双视角本身是否有效
2. 辅助监督是否有效
3. shared backbone 是否比双 backbone 更稳
4. full branch 强增强、roi branch 弱增强 是否优于两边都强增强

这一步很重要，否则即使新模型涨了，也不知道到底为什么涨。

### 16.3 第三优先级：把阈值调优正式化

现在仓库里很多增益都依赖阈值调优。

所以后续必须把它从“最后补一刀”升级成正式流程：

- 单独记录默认阈值结果
- 单独记录最优阈值结果
- 后面如果补 val，就在 val 上选阈值

### 16.4 第四优先级：继续保留 multi-seed

任何新 Transformer 主线，如果只跑一个 seed，信息量不够。

建议至少：

- 跑 `3 seeds`
- 报 mean/std

否则很难判断是真改进还是随机波动。

---

## 17. 如果真的要冲四个指标 80%，还必须做什么

下面这些不是“可选增强项”，而是我认为 **80% 目标的必要条件**。

### 17.1 做患者级验证或独立验证集

当前流程里：

- best epoch 用 test 选
- threshold 也在 test 上选

这会让结果偏乐观，也会误导模型决策方向。

如果后续真要冲 80%，第一件事不是换模型，而是：

- 做一个可信的 val

### 17.2 做错误样本审计

必须把以下样本单独拉出来看：

- 高置信度错判 benign
- 高置信度错判 no_tumor
- 不同 seed 预测分歧大的样本
- stacking 能纠正、单模型纠正不了的样本

这一步的目标是回答：

- 错误到底来自图像不可分，还是标签边界不稳，还是标注信息没被模型用到

### 17.3 重点补 benign 样本

如果新增数据预算有限，优先级应该是：

- 补 `benign`
- 特别补边界样本和多样性样本

因为当前瓶颈几乎都集中在少数类。

### 17.4 引入临床信息

如果真要 80%，我认为临床信息几乎是必选项。

因为现有文档已经给出一个非常清楚的暗示：

- 图像差异可能不足以支撑稳定区分
- 任务差异更接近病理语义

这类任务最适合的不是“只靠图像再堆模型”，而是：

- 图像 + 临床 + 标注联合建模

---

## 18. 最终建议：怎么推进，才不是盲目冲分

### 18.1 现在立刻该做的

1. 跑新的双视角 Transformer 主线
2. 做关键 ablation
3. 记录默认阈值和最优阈值
4. 至少跑 3 个 seed

### 18.2 接下来一周内该做的

1. 建立 val 机制
2. 做错例审计
3. 判断 aux supervision 是否真有效
4. 判断 full/roi 双视角是否真带来互补

### 18.3 真想冲 80%，必须同步做的

1. benign 数据扩充
2. 标签质量审计
3. 临床信息接入
4. 患者级验证

---

## 19. 一句话版

如果让我只给一句最硬的建议，我会写：

> 现在最合理的路线不是再横向换模型，而是把 `SwinV2` 这条 Transformer 主线升级成“全图上下文 + ROI 细节 + 标注辅助监督”的双视角模型；但如果目标是把四个指标都稳定推到 `80%`，那模型只是其中一部分，真正的突破口仍然在验证流程、数据质量和少数类样本补强。

