
from pathlib import Path
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    confusion_matrix
)


# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "apachejit_processed.csv"

MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

MODEL_PATH = MODELS_DIR / "final_model.joblib"

FEATURE_COLUMNS = [
    "la", "ld", "nf", "nd", "ns", "ent",
    "ndev", "age", "nuc", "aexp", "arexp", "asexp"
]

TARGET_COLUMN = "buggy"
DATE_COLUMN = "author_date"
ID_COLUMN = "commit_id"

RANDOM_STATE = 42
VALIDATION_FRACTION = 0.20
PREDICTION_THRESHOLD = 0.50


# ==========================================
# 2. LOAD AND CHECK DATA
# ==========================================

def load_data():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    required = (
        FEATURE_COLUMNS
        + [TARGET_COLUMN, DATE_COLUMN, ID_COLUMN]
    )

    missing = set(required) - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    X = df[FEATURE_COLUMNS].apply(
        pd.to_numeric, errors="raise"
    ).astype(float)

    y = pd.to_numeric(
        df[TARGET_COLUMN], errors="raise"
    ).astype(int)

    raw_dates = pd.to_numeric(
        df[DATE_COLUMN],
        errors="coerce"
    )

    if raw_dates.notna().all():
        dates = pd.to_datetime(
            raw_dates,
            unit="s",
            utc=True,
            errors="coerce"
        )
    else:
        dates = pd.to_datetime(
            df[DATE_COLUMN],
            utc=True,
            errors="coerce"
        )

    if X.isna().any().any():
        raise ValueError(
            "Missing feature values found. "
            "Investigate preprocessing first."
        )

    if not np.isfinite(X.to_numpy()).all():
        raise ValueError(
            "Infinite feature values found."
        )

    if set(y.unique()) != {0, 1}:
        raise ValueError(
            f"Expected target labels 0 and 1, got {set(y.unique())}"
        )

    if dates.isna().any():
        raise ValueError(
            f"{dates.isna().sum()} invalid author_date values. "
            "Fix dates before temporal splitting."
        )

    print("\nDataset loaded.")
    print(f"Dataset shape: {df.shape}")
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    print("\nTarget distribution:")
    print(y.value_counts().sort_index())

    print("\nProject count:", df["project"].nunique())

    return df, X, y, dates


# ==========================================
# 3. CHRONOLOGICAL TRAIN / VALIDATION SPLIT
# ==========================================

def temporal_split(df, X, y, dates):

    sorted_dates = dates.sort_values(
        kind="mergesort"
    )

    split_position = int(
        len(df) * (1 - VALIDATION_FRACTION)
    )

    cutoff_date = sorted_dates.iloc[split_position]

    train_mask = dates < cutoff_date
    validation_mask = dates >= cutoff_date

    if train_mask.sum() == 0 or validation_mask.sum() == 0:
        raise ValueError(
            "Temporal split produced an empty partition."
        )

    X_train = X.loc[train_mask]
    X_val = X.loc[validation_mask]

    y_train = y.loc[train_mask]
    y_val = y.loc[validation_mask]

    if set(y_train.unique()) != {0, 1}:
        raise ValueError(
            "Training partition must contain both classes."
        )

    if set(y_val.unique()) != {0, 1}:
        raise ValueError(
            "Validation partition must contain both classes. "
            "Review the temporal cutoff."
        )

    # Prevent identical commit IDs appearing in both partitions.
    train_ids = set(
        df.loc[train_mask, ID_COLUMN].dropna().astype(str)
    )

    val_ids = set(
        df.loc[validation_mask, ID_COLUMN].dropna().astype(str)
    )

    overlap = train_ids.intersection(val_ids)

    if overlap:
        raise ValueError(
            f"{len(overlap)} commit IDs occur in both partitions."
        )

    print("\n=== TEMPORAL SPLIT ===")

    print(f"Cutoff: {cutoff_date}")
    print(f"Training rows: {len(X_train)}")
    print(f"Validation rows: {len(X_val)}")

    print("\nTraining class distribution:")
    print(y_train.value_counts().sort_index())

    print("\nValidation class distribution:")
    print(y_val.value_counts().sort_index())

    print("\nValidation date range:")
    print(dates.loc[validation_mask].min())
    print(dates.loc[validation_mask].max())

    return X_train, X_val, y_train, y_val, cutoff_date


# ==========================================
# 4. DEFINE THREE CANDIDATE MODELS
# ==========================================

def create_models():

    return {
        "Logistic Regression": LogisticRegression(
            max_iter=3000,
            random_state=RANDOM_STATE
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_features="sqrt",
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=RANDOM_STATE
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=2,
            min_samples_leaf=5,
            subsample=0.8,
            random_state=RANDOM_STATE
        )
    }


# ==========================================
# 5. CREATE LEAKAGE-SAFE PIPELINES
# ==========================================

def create_pipeline(model_name, classifier):

    steps = [
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]

    # Standardization is useful for Logistic Regression.
    # Tree models do not require feature scaling.

    if model_name == "Logistic Regression":
        steps.append(
            ("scaler", StandardScaler())
        )

    steps.append(
        ("classifier", classifier)
    )

    return Pipeline(steps)


# ==========================================
# 6. TRAIN AND EVALUATE
# ==========================================

def train_and_evaluate(
    models,
    X_train,
    X_val,
    y_train,
    y_val
):

    results = []
    fitted_models = {}

    for model_name, classifier in models.items():

        print("\n" + "=" * 55)
        print(f"TRAINING: {model_name}")
        print("=" * 55)

        pipeline = create_pipeline(
            model_name, classifier
        )

        # Every learned preprocessing step is fitted
        # only using the training partition.
        pipeline.fit(X_train, y_train)

        probabilities = pipeline.predict_proba(
            X_val
        )[:, 1]

        predictions = (
            probabilities >= PREDICTION_THRESHOLD
        ).astype(int)

        cm = confusion_matrix(
            y_val,
            predictions,
            labels=[0, 1]
        )

        tn, fp, fn, tp = cm.ravel()

        metrics = {
            "model": model_name,

            "accuracy": accuracy_score(
                y_val, predictions
            ),

            "precision": precision_score(
                y_val, predictions, zero_division=0
            ),

            "recall": recall_score(
                y_val, predictions, zero_division=0
            ),

            "f1": f1_score(
                y_val, predictions, zero_division=0
            ),

            "roc_auc": roc_auc_score(
                y_val, probabilities
            ),

            "pr_auc": average_precision_score(
                y_val, probabilities
            ),

            "balanced_accuracy": balanced_accuracy_score(
                y_val, predictions
            ),

            "mcc": matthews_corrcoef(
                y_val, predictions
            ),

            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        }

        results.append(metrics)
        fitted_models[model_name] = pipeline

        print(f"\nAccuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-score:  {metrics['f1']:.4f}")
        print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
        print(f"PR-AUC:    {metrics['pr_auc']:.4f}")

        print("\nConfusion Matrix:")
        print(cm)

    return results, fitted_models


# ==========================================
# 7. SELECT BEST MODEL
# ==========================================

def select_best_model(results, fitted_models):

    # F1 is our primary selection metric.
    # PR-AUC is the tie-breaker.

    ranked = sorted(
        results,
        key=lambda r: (
            r["f1"],
            r["pr_auc"],
            r["roc_auc"]
        ),
        reverse=True
    )

    best = ranked[0]
    best_name = best["model"]

    print("\n=== MODEL COMPARISON ===")

    comparison = pd.DataFrame(ranked)

    print(
        comparison[
            ["model", "precision", "recall",
             "f1", "roc_auc", "pr_auc"]
        ].to_string(index=False)
    )

    print(f"\nSelected model: {best_name}")

    return best_name, best, fitted_models[best_name]


# ==========================================
# 8. SAVE REPORT
# ==========================================

def save_report(
    results,
    best_name,
    best_metrics,
    cutoff_date,
    train_rows,
    validation_rows
):

    comparison = pd.DataFrame(results)

    REPORTS_DIR.mkdir(
        parents=True, exist_ok=True
    )

    comparison.to_csv(
        REPORTS_DIR / "model_comparison.csv",
        index=False
    )

    ranked = comparison.sort_values(
        by=["f1", "pr_auc", "roc_auc"],
        ascending=False
    )

    markdown_table = ranked[
        [
            "model",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "pr_auc",
            "balanced_accuracy",
            "mcc"
        ]
    ].to_markdown(
        index=False,
        floatfmt=".4f"
    )

    report = f"""
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

- Training rows: {train_rows}
- Validation rows: {validation_rows}
- Cutoff date: {cutoff_date}
- Official future test datasets: Not used.

The author_date field was used only to construct
the temporal split. It was excluded from model inputs.

All preprocessing was fitted within the training partition.

## Model comparison

{markdown_table}

## Selected model

**{best_name}**

Selection criteria:

1. Highest validation F1-score.
2. PR-AUC as the first tie-breaker.
3. ROC-AUC as the second tie-breaker.

### Selected model metrics

- Precision: {best_metrics['precision']:.4f}
- Recall: {best_metrics['recall']:.4f}
- F1-score: {best_metrics['f1']:.4f}
- ROC-AUC: {best_metrics['roc_auc']:.4f}
- PR-AUC: {best_metrics['pr_auc']:.4f}
- Balanced Accuracy: {best_metrics['balanced_accuracy']:.4f}
- MCC: {best_metrics['mcc']:.4f}

### Confusion matrix

| | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | {best_metrics['tn']} | {best_metrics['fp']} |
| Actual 1 | {best_metrics['fn']} | {best_metrics['tp']} |

## Final evaluation restriction

The official ApacheJIT future test datasets must remain
untouched until final evaluation.

Validation metrics above are not final test metrics.
"""

    with open(
        REPORTS_DIR / "model_evaluation.md",
        "w",
        encoding="utf-8"
    ) as file:

        file.write(report)


# ==========================================
# 9. REFIT SELECTED MODEL ON ALL TRAINING DATA
# ==========================================

def save_final_model(
    best_name,
    selected_pipeline,
    best_metrics,
    X,
    y
):

    # The model was selected using the temporal
    # validation partition. Now refit a fresh copy
    # on all available historical development data.

    final_pipeline = create_pipeline(
        best_name,
        selected_pipeline.named_steps["classifier"]
    )

    final_pipeline.fit(X, y)

    MODELS_DIR.mkdir(
        parents=True, exist_ok=True
    )

    artifact = {
        "pipeline": final_pipeline,
        "model_name": best_name,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "threshold": PREDICTION_THRESHOLD,
        "training_rows": len(X),
        "selected_validation_metrics": best_metrics,
        "created_at": datetime.now(
            timezone.utc
        ).isoformat()
    }

    joblib.dump(artifact, MODEL_PATH)

    print("\nFinal model saved:")
    print(MODEL_PATH)

    print(
        "\nThe saved artifact contains the complete "
        "preprocessing and prediction pipeline."
    )


# ==========================================
# 10. MAIN EXECUTION
# ==========================================

def main():

    print("\nAPACHEJIT SOFTWARE DEFECT PREDICTION")
    print("ML Training Pipeline")
    print("=" * 55)

    df, X, y, dates = load_data()

    X_train, X_val, y_train, y_val, cutoff = (
        temporal_split(df, X, y, dates)
    )

    models = create_models()

    results, fitted_models = train_and_evaluate(
        models,
        X_train,
        X_val,
        y_train,
        y_val
    )

    best_name, best_metrics, best_pipeline = (
        select_best_model(
            results,
            fitted_models
        )
    )

    save_report(
        results,
        best_name,
        best_metrics,
        cutoff,
        len(X_train),
        len(X_val)
    )

    save_final_model(
        best_name,
        best_pipeline,
        best_metrics,
        X,
        y
    )

    print("\n" + "=" * 55)
    print("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 55)

    print(f"\nBest model: {best_name}")
    print(f"Validation F1: {best_metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
