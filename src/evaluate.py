
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import shap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay
)

from train import (
    load_data,
    temporal_split,
    create_models,
    create_pipeline,
    MODEL_PATH,
    REPORTS_DIR,
    PREDICTION_THRESHOLD
)


ROOT = Path(__file__).resolve().parents[1]

SHAP_SAMPLE_SIZE = 300
SHAP_BACKGROUND_SIZE = 100


# ==========================================
# 1. VALIDATION EVALUATION
# ==========================================

def evaluate_selected_model():

    print("\n=== VALIDATION EVALUATION ===")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Train the models first. final_model.joblib not found."
        )

    artifact = joblib.load(MODEL_PATH)

    model_name = artifact["model_name"]
    threshold = artifact.get(
        "threshold", PREDICTION_THRESHOLD
    )

    print(f"Selected model: {model_name}")

    df, X, y, dates = load_data()

    X_train, X_val, y_train, y_val, cutoff = (
        temporal_split(df, X, y, dates)
    )

    models = create_models()

    if model_name not in models:
        raise ValueError(
            f"Unknown model in artifact: {model_name}"
        )

    # Re-create the selected candidate.
    # Fit ONLY on the historical training partition.
    validation_pipeline = create_pipeline(
        model_name,
        models[model_name]
    )

    validation_pipeline.fit(
        X_train,
        y_train
    )

    probabilities = validation_pipeline.predict_proba(
        X_val
    )[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    cm = confusion_matrix(
        y_val,
        predictions,
        labels=[0, 1]
    )

    metrics = {
        "model": model_name,
        "cutoff_date": str(cutoff),
        "training_rows": int(len(X_train)),
        "validation_rows": int(len(X_val)),
        "threshold": float(threshold),

        "accuracy": float(
            accuracy_score(y_val, predictions)
        ),

        "precision": float(
            precision_score(
                y_val, predictions, zero_division=0
            )
        ),

        "recall": float(
            recall_score(
                y_val, predictions, zero_division=0
            )
        ),

        "f1": float(
            f1_score(
                y_val, predictions, zero_division=0
            )
        ),

        "roc_auc": float(
            roc_auc_score(y_val, probabilities)
        ),

        "pr_auc": float(
            average_precision_score(y_val, probabilities)
        ),

        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_val, predictions
            )
        ),

        "mcc": float(
            matthews_corrcoef(y_val, predictions)
        ),

        "confusion_matrix": cm.tolist()
    }

    print("\n=== VALIDATION METRICS ===")

    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")

    print("\n=== CLASSIFICATION REPORT ===")

    report = classification_report(
        y_val,
        predictions,
        target_names=[
            "Non-defective",
            "Defective"
        ],
        zero_division=0
    )

    print(report)

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        REPORTS_DIR / "evaluation_metrics.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            metrics,
            file,
            indent=4
        )

    with open(
        REPORTS_DIR / "classification_report.txt",
        "w",
        encoding="utf-8"
    ) as file:
        file.write(report)

    return (
        df,
        X,
        y,
        metrics,
        y_val,
        predictions,
        probabilities
    )


# ==========================================
# 2. CONFUSION MATRIX
# ==========================================

def create_confusion_matrix(y_true, predictions):

    print("\nGenerating confusion matrix...")

    fig, ax = plt.subplots(figsize=(7, 6))

    display = ConfusionMatrixDisplay.from_predictions(
        y_true,
        predictions,
        labels=[0, 1],
        display_labels=[
            "Non-defective",
            "Defective"
        ],
        cmap="Blues",
        ax=ax
    )

    ax.set_title(
        "ApacheJIT - Confusion Matrix"
    )

    fig.tight_layout()

    fig.savefig(
        REPORTS_DIR / "confusion_matrix.png",
        dpi=300
    )

    plt.close(fig)


# ==========================================
# 3. ROC CURVE
# ==========================================

def create_roc_curve(y_true, probabilities):

    print("Generating ROC curve...")

    fig, ax = plt.subplots(figsize=(7, 6))

    RocCurveDisplay.from_predictions(
        y_true,
        probabilities,
        ax=ax,
        name="Selected Model"
    )

    ax.set_title(
        "ApacheJIT - ROC Curve"
    )

    fig.tight_layout()

    fig.savefig(
        REPORTS_DIR / "roc_curve.png",
        dpi=300
    )

    plt.close(fig)


# ==========================================
# 4. PRECISION-RECALL CURVE
# ==========================================

def create_precision_recall_curve(
    y_true,
    probabilities
):

    print("Generating Precision-Recall curve...")

    fig, ax = plt.subplots(figsize=(7, 6))

    PrecisionRecallDisplay.from_predictions(
        y_true,
        probabilities,
        ax=ax,
        name="Selected Model"
    )

    ax.set_title(
        "ApacheJIT - Precision-Recall Curve"
    )

    fig.tight_layout()

    fig.savefig(
        REPORTS_DIR / "precision_recall_curve.png",
        dpi=300
    )

    plt.close(fig)


# ==========================================
# 5. SHAP EXPLAINABILITY
# ==========================================

def generate_shap_explanations(X):

    print("\n=== SHAP EXPLAINABILITY ===")

    artifact = joblib.load(MODEL_PATH)

    final_pipeline = artifact["pipeline"]

    classifier = final_pipeline.named_steps[
        "classifier"
    ]

    feature_names = artifact["feature_columns"]

    rng = np.random.RandomState(42)

    sample_size = min(
        SHAP_SAMPLE_SIZE,
        len(X)
    )

    background_size = min(
        SHAP_BACKGROUND_SIZE,
        len(X)
    )

    sample = X.sample(
        n=sample_size,
        random_state=42
    )

    background = X.sample(
        n=background_size,
        random_state=43
    )

    # Transform using the SAME preprocessing
    # fitted inside our saved model pipeline.
    transformed_sample = (
        final_pipeline[:-1].transform(sample)
    )

    transformed_background = (
        final_pipeline[:-1].transform(background)
    )

    print("Building SHAP explainer...")

    explainer = shap.Explainer(
        classifier,
        transformed_background
    )

    explanation = explainer(
        transformed_sample
    )

    values = np.asarray(
        explanation.values
    )

    base_values = np.asarray(
        explanation.base_values
    )

    # For binary models that return SHAP values
    # for both output classes, select class 1.
    if values.ndim == 3:

        values = values[:, :, 1]

        if base_values.ndim == 2:
            base_values = base_values[:, 1]

    shap_explanation = shap.Explanation(
        values=values,
        base_values=base_values,
        data=transformed_sample,
        feature_names=feature_names
    )

    # ======================================
    # SHAP FEATURE IMPORTANCE
    # ======================================

    importance = pd.DataFrame({
        "feature": feature_names,
        "mean_absolute_shap": np.abs(
            values
        ).mean(axis=0)
    })

    importance = importance.sort_values(
        "mean_absolute_shap",
        ascending=False
    )

    importance.to_csv(
        REPORTS_DIR / "shap_feature_importance.csv",
        index=False
    )

    print("\n=== SHAP FEATURE IMPORTANCE ===")
    print(importance.to_string(index=False))

    # ======================================
    # SHAP BAR PLOT
    # ======================================

    plt.figure(figsize=(10, 7))

    shap.plots.bar(
        shap_explanation,
        max_display=12,
        show=False
    )

    plt.tight_layout()

    plt.savefig(
        REPORTS_DIR / "shap_feature_importance.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ======================================
    # SHAP BEESWARM PLOT
    # ======================================

    plt.figure(figsize=(11, 8))

    shap.plots.beeswarm(
        shap_explanation,
        max_display=12,
        show=False
    )

    plt.tight_layout()

    plt.savefig(
        REPORTS_DIR / "shap_summary.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("\nSHAP explanations generated.")

    print(
        "Sample explained:",
        sample_size
    )

    return importance


# ==========================================
# 6. FINAL EVALUATION REPORT
# ==========================================

def generate_evaluation_report(
    metrics,
    importance
):

    print("\nGenerating evaluation report...")

    feature_rows = "\n".join(
        f"| {row.feature} | "
        f"{row.mean_absolute_shap:.6f} |"
        for row in importance.itertuples()
    )

    cm = metrics["confusion_matrix"]

    report = f"""
# ApacheJIT Model Evaluation

## Selected model

{metrics['model']}

## Validation strategy

Chronological historical holdout.

- Training rows: {metrics['training_rows']}
- Validation rows: {metrics['validation_rows']}
- Cutoff date: {metrics['cutoff_date']}
- Prediction threshold: {metrics['threshold']}

Only the 12 approved software metrics were used as
model input. The date and commit ID were used for
validation integrity, not prediction.

## Validation metrics

| Metric | Result |
|---|---:|
| Accuracy | {metrics['accuracy']:.4f} |
| Precision | {metrics['precision']:.4f} |
| Recall | {metrics['recall']:.4f} |
| F1-score | {metrics['f1']:.4f} |
| ROC-AUC | {metrics['roc_auc']:.4f} |
| PR-AUC | {metrics['pr_auc']:.4f} |
| Balanced Accuracy | {metrics['balanced_accuracy']:.4f} |
| Matthews Correlation | {metrics['mcc']:.4f} |

## Confusion Matrix

| | Predicted Non-defective | Predicted Defective |
|---|---:|---:|
| Actual Non-defective | {cm[0][0]} | {cm[0][1]} |
| Actual Defective | {cm[1][0]} | {cm[1][1]} |

## SHAP Feature Importance

Mean absolute SHAP values, calculated from
a sample of the historical development dataset.

| Feature | Mean absolute SHAP |
|---|---:|
{feature_rows}

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
"""

    with open(
        REPORTS_DIR / "model_evaluation.md",
        "w",
        encoding="utf-8"
    ) as file:

        file.write(report)


# ==========================================
# 7. MAIN EXECUTION
# ==========================================

def main():

    (
        df,
        X,
        y,
        metrics,
        y_val,
        predictions,
        probabilities
    ) = evaluate_selected_model()

    create_confusion_matrix(
        y_val,
        predictions
    )

    create_roc_curve(
        y_val,
        probabilities
    )

    create_precision_recall_curve(
        y_val,
        probabilities
    )

    importance = generate_shap_explanations(
        X
    )

    generate_evaluation_report(
        metrics,
        importance
    )

    print("\n" + "=" * 55)
    print("EVALUATION AND SHAP COMPLETED")
    print("=" * 55)

    print("\nGenerated reports:")
    for path in sorted(REPORTS_DIR.iterdir()):
        if path.is_file():
            print(f" - {path.name}")


if __name__ == "__main__":
    main()
