
# ApacheJIT Model Evaluation Report

## Dataset

- Dataset: ApacheJIT
- Processed rows: 44,834 (expected)
- Input features: 12
- Target: buggy
- Class 0: Non-defective
- Class 1: Defective

## Validation strategy

Chronological holdout.

- Training rows: 35867
- Validation rows: 8967
- Cutoff date: 2015-07-25 11:59:30+00:00
- Official future test datasets: Not used.

The author_date field was used only to construct
the temporal split. It was excluded from model inputs.

All preprocessing was fitted within the training partition.

## Model comparison

| model               |   accuracy |   precision |   recall |     f1 |   roc_auc |   pr_auc |   balanced_accuracy |    mcc |
|:--------------------|-----------:|------------:|---------:|-------:|----------:|---------:|--------------------:|-------:|
| Gradient Boosting   |     0.7475 |      0.7349 |   0.7862 | 0.7597 |    0.8204 |   0.8133 |              0.7469 | 0.4956 |
| Random Forest       |     0.7422 |      0.7272 |   0.7873 | 0.7561 |    0.8179 |   0.8134 |              0.7415 | 0.4853 |
| Logistic Regression |     0.7151 |      0.7564 |   0.6469 | 0.6974 |    0.7757 |   0.7766 |              0.7161 | 0.4360 |

## Selected model

**Gradient Boosting**

Selection criteria:

1. Highest validation F1-score.
2. PR-AUC as the first tie-breaker.
3. ROC-AUC as the second tie-breaker.

### Selected model metrics

- Precision: 0.7349
- Recall: 0.7862
- F1-score: 0.7597
- ROC-AUC: 0.8204
- PR-AUC: 0.8133
- Balanced Accuracy: 0.7469
- MCC: 0.4956

### Confusion matrix

| | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | 3125 | 1291 |
| Actual 1 | 973 | 3578 |

## Final evaluation restriction

The official ApacheJIT future test datasets must remain
untouched until final evaluation.

Validation metrics above are not final test metrics.
