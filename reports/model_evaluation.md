
# ApacheJIT Model Evaluation

## Selected model

Gradient Boosting

## Validation strategy

Chronological historical holdout.

- Training rows: 35867
- Validation rows: 8967
- Cutoff date: 1970-01-01 00:00:01.437825570+00:00
- Prediction threshold: 0.5

Only the 12 approved software metrics were used as
model input. The date and commit ID were used for
validation integrity, not prediction.

## Validation metrics

| Metric | Result |
|---|---:|
| Accuracy | 0.7475 |
| Precision | 0.7349 |
| Recall | 0.7862 |
| F1-score | 0.7597 |
| ROC-AUC | 0.8204 |
| PR-AUC | 0.8133 |
| Balanced Accuracy | 0.7469 |
| Matthews Correlation | 0.4956 |

## Confusion Matrix

| | Predicted Non-defective | Predicted Defective |
|---|---:|---:|
| Actual Non-defective | 3125 | 1291 |
| Actual Defective | 973 | 3578 |

## SHAP Feature Importance

Mean absolute SHAP values, calculated from
a sample of the historical development dataset.

| Feature | Mean absolute SHAP |
|---|---:|
| la | 0.761154 |
| ld | 0.196373 |
| age | 0.169455 |
| nuc | 0.157800 |
| asexp | 0.086395 |
| ns | 0.085511 |
| ndev | 0.054876 |
| nf | 0.051362 |
| arexp | 0.041912 |
| aexp | 0.015357 |
| ent | 0.001053 |
| nd | 0.000024 |

## Generated visualizations

- confusion_matrix.png
- roc_curve.png
- precision_recall_curve.png
- shap_feature_importance.png
- shap_summary.png

## Important limitations

These are historical validation results, not official
future-test results.

The official ApacheJIT future test datasets remain
reserved for final evaluation.

SHAP importance describes model behavior and should
not be interpreted as proof of causation.
