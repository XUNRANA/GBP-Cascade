# GBP-Cascade Task 2 自主实验协议

胆囊息肉超声图像分类: 良性肿瘤(benign) vs 非肿瘤性息肉(no_tumor) 二分类.

## Setup

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar24`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current state.
3. **Read the in-scope files**:
   - `prepare_task2.py` — fixed constants, datasets, transforms, evaluation. **Do not modify.**
   - `train_task2.py` — the file you modify. Model, hyperparameters, training loop.
4. **Verify data exists**: Check that `../0322dataset/` contains `task_2_train.xlsx`, `task_2_test.xlsx`, and image directories (`benign/`, `no_tumor/`).
5. **Initialize results.tsv**: Create `results.tsv` with just the header row.
6. **Confirm and go**.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. Launch: `python train_task2.py`

**What you CAN do:**
- Modify `train_task2.py` — this is the only file you edit. Everything is fair game: model selection, hyperparameters, training loop, augmentation strategy, loss function, optimizer configuration.

**What you CANNOT do:**
- Modify `prepare_task2.py`. It is read-only. It contains the fixed evaluation, datasets, transforms, and constants.
- Install new packages. Only use what's available: `timm`, `torch`, `sklearn`, `PIL`, `numpy`, `pandas`.
- Modify the evaluation harness. `evaluate_model()` in `prepare_task2.py` is the ground truth metric.

**The goal is simple: get the highest `f1_at_threshold`.** This is the F1(macro) after optimal threshold search on P(benign). Higher is better.

**Current best baseline: `f1_at_threshold ≈ 0.624`** (SwinV2-Tiny + full 4ch + strong aug + balanced sampler + weighted mixup).

**VRAM** is a soft constraint. Some increase is acceptable for meaningful gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Removing something and getting equal or better results is a great outcome.

**The first run**: Your very first run should always be to establish the baseline, so run the training script as-is.

## Known constraints and tips

**Data constraints (critical):**
- Only 309 benign vs 920 no_tumor training images (1:3 class imbalance)
- Lesion area difference between classes is only ~0.7% — extremely subtle visual difference
- 523 test images (129 benign, 394 no_tumor)

**Proven effective:**
- 4-channel input (RGB + lesion mask) consistently outperforms 3-channel (+2-3% F1)
- StrongSyncTransform > SyncTransform for reducing overfitting
- WeightedRandomSampler helps with class imbalance
- Threshold search improves F1 by 1-2% over default 0.5

**Known pitfalls:**
- **Mixup bug**: When using Mixup with soft labels, the loss MUST include class weights. The code already has this fix — do not remove it.
- **Overfitting**: Models reach 93%+ train acc while test F1 stays ~60%. Aggressive regularization is key.
- **Evaluation**: Models are evaluated every EVAL_INTERVAL epochs and the best checkpoint is kept.

### High-priority search directions
1. **Data augmentation params** — CutMix/Mixup alpha, crop scale, color jitter range, noise level
2. **Class balancing** — sampler weights, focal loss gamma, class weight scaling factor
3. **Learning rate schedule** — warmup duration, decay shape, backbone/head LR ratio
4. **Model selection** — efficientnet, convnext, swin, deit, eva, maxvit (any timm model)
5. **Regularization** — dropout, weight decay, label smoothing, stochastic depth
6. **Input representation** — ROI crop padding ratio, mask channel weight, image size
7. **Training tricks** — EMA, gradient accumulation, longer training, cosine restarts

### Timm models available (tested or promising)
- `swinv2_tiny_window8_256` (current best)
- `convnext_tiny.fb_in22k`, `convnext_small.fb_in22k`
- `resnet34`, `resnet50`
- `maxvit_tiny_tf_256`
- `deit3_small_patch16_224`
- `efficientnet_b0`, `efficientnet_b2`, `efficientnet_b3`
- `eva02_small_patch14_336.mim_in22k_ft_in1k`
- `caformer_s18.sail_in22k_ft_in1k`
- Any other timm model that accepts `pretrained=True` and `num_classes=2`

## Output format

The script prints a summary block at the end:

```
---
f1_at_threshold:       0.624000
best_threshold:        0.530
f1_macro:              0.617300
accuracy:              0.780000
...
peak_vram_mb:          4500.2
num_params_M:          28.3
```

Extract the key metric: `grep "^f1_at_threshold:" run.log`
Extract memory: `grep "^peak_vram_mb:" run.log`

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated).

The TSV has a header row and 5 columns:

```
commit	f1_at_threshold	peak_vram_mb	status	description
```

1. `commit` — git commit hash (short, 7 chars)
2. `f1_at_threshold` — the primary metric (e.g. 0.624000). Use 0.000000 for crashes.
3. `peak_vram_mb` — peak VRAM in MB (e.g. 4500.2). Use 0.0 for crashes.
4. `status` — `keep`, `discard`, or `crash`
5. `description` — short text describing what this experiment tried

Example:

```
commit	f1_at_threshold	peak_vram_mb	status	description
a1b2c3d	0.624000	4500.2	keep	baseline (swinv2 + full4ch + strong aug + balanced sampler)
b2c3d4e	0.630100	4600.0	keep	increased dropout to 0.4 + reduced LR
c3d4e5f	0.618000	3200.0	discard	switched to resnet50
d4e5f6g	0.000000	0.0	crash	tried convnext_large (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar24`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Modify `train_task2.py` with an experimental idea
3. `git commit -m "description of change"`
4. Run the experiment: `python train_task2.py > run.log 2>&1`
5. Read out the results: `grep "^f1_at_threshold:\|^peak_vram_mb:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the stack trace and attempt a fix.
7. Record the results in results.tsv (NOTE: do not commit results.tsv, leave it untracked)
8. If `f1_at_threshold` improved (higher), you "advance" the branch, keeping the git commit
9. If `f1_at_threshold` is equal or worse, you `git reset --hard HEAD~1` to discard

**Timeout**: Each experiment should take < 10 minutes. If a run exceeds 15 minutes, kill it and treat as failure.

**Crashes**: If it's a typo or easy fix, fix it and re-run. If fundamentally broken, skip it, log "crash", and move on.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human if you should continue. The human might be asleep. You are autonomous. If you run out of ideas: re-read the files, try combining previous near-misses, try more radical changes, try different model families. The loop runs until the human interrupts you, period.

As a rough estimate: each experiment takes ~5-8 minutes, so you can run ~8-12 per hour, or ~60-100 over 8 hours of sleep.
