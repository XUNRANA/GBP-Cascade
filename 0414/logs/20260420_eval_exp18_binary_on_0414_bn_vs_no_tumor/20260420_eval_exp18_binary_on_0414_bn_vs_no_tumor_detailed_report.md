# 20260420_eval_exp18_binary_on_0414_bn_vs_no_tumor

## Dataset
- Binary eval samples: 342
- Benign: 114
- No_tumor: 228

## Probability Quality
- ROC-AUC (no_tumor positive): 0.7789
- PR-AUC (no_tumor positive): 0.8445
- ECE (P(no_tumor)): 0.0463
- Brier (P(no_tumor)): 0.1691

## Threshold Comparison
| Setting | Threshold | Acc | Precision(macro) | Recall(macro) | F1(macro) |
|---|---:|---:|---:|---:|---:|
| Default | 0.500 | 0.7632 | 0.7427 | 0.6974 | 0.7095 |
| Best-F1 | 0.470 | 0.7661 | 0.7413 | 0.7105 | 0.7206 |
| Policy(miss<=10%) | 0.184 | 0.5965 | 0.6748 | 0.6732 | 0.5965 |
| Policy(miss<=05%) | 0.144 | 0.3918 | 0.5913 | 0.5329 | 0.3528 |

## Default Threshold Classification Report
```text
              precision    recall  f1-score   support

      benign     0.7037    0.5000    0.5846       114
    no_tumor     0.7816    0.8947    0.8344       228

    accuracy                         0.7632       342
   macro avg     0.7427    0.6974    0.7095       342
weighted avg     0.7556    0.7632    0.7511       342
```

## Best-F1 Threshold Classification Report
```text
              precision    recall  f1-score   support

      benign     0.6889    0.5439    0.6078       114
    no_tumor     0.7937    0.8772    0.8333       228

    accuracy                         0.7661       342
   macro avg     0.7413    0.7105    0.7206       342
weighted avg     0.7587    0.7661    0.7582       342
```

## Policy(miss<=10%) Classification Report
```text
              precision    recall  f1-score   support

      benign     0.4478    0.9035    0.5988       114
    no_tumor     0.9018    0.4430    0.5941       228

    accuracy                         0.5965       342
   macro avg     0.6748    0.6732    0.5965       342
weighted avg     0.7505    0.5965    0.5957       342
```

- Policy details: benign_miss_rate=9.65%, benign_recall=90.35%, no_tumor_recall=44.30%, no_tumor_precision=90.18%, constraint_satisfied=True

## Policy(miss<=05%) Classification Report
```text
              precision    recall  f1-score   support

      benign     0.3494    0.9561    0.5117       114
    no_tumor     0.8333    0.1096    0.1938       228

    accuracy                         0.3918       342
   macro avg     0.5913    0.5329    0.3528       342
weighted avg     0.6720    0.3918    0.2998       342
```

- Policy details: benign_miss_rate=4.39%, benign_recall=95.61%, no_tumor_recall=10.96%, no_tumor_precision=83.33%, constraint_satisfied=True

## Artifacts
- Binary eval excel: `/data1/ouyangxinglong/GBP-Cascade/0414/logs/20260420_eval_exp18_binary_on_0414_bn_vs_no_tumor/0414_binary_eval_test.xlsx`
- Per-case probabilities: `/data1/ouyangxinglong/GBP-Cascade/0414/logs/20260420_eval_exp18_binary_on_0414_bn_vs_no_tumor/20260420_eval_exp18_binary_on_0414_bn_vs_no_tumor_probs.csv`
- Confusion matrix (default): `/data1/ouyangxinglong/GBP-Cascade/0414/logs/20260420_eval_exp18_binary_on_0414_bn_vs_no_tumor/20260420_eval_exp18_binary_on_0414_bn_vs_no_tumor_confusion_matrix_default.csv`
- Reliability bins csv: `/data1/ouyangxinglong/GBP-Cascade/0414/logs/20260420_eval_exp18_binary_on_0414_bn_vs_no_tumor/20260420_eval_exp18_binary_on_0414_bn_vs_no_tumor_reliability_no_tumor.csv`
- Reliability diagram png: `/data1/ouyangxinglong/GBP-Cascade/0414/logs/20260420_eval_exp18_binary_on_0414_bn_vs_no_tumor/20260420_eval_exp18_binary_on_0414_bn_vs_no_tumor_reliability_no_tumor.png`
- Metrics json: `/data1/ouyangxinglong/GBP-Cascade/0414/logs/20260420_eval_exp18_binary_on_0414_bn_vs_no_tumor/20260420_eval_exp18_binary_on_0414_bn_vs_no_tumor_metrics.json`
