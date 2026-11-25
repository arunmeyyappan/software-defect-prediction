
from pathlib import Path
import pandas as pd

# Project configuration
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "apachejit_processed.csv"
)

FEATURE_COLUMNS = [
    "la", "ld", "nf", "nd", "ns", "ent",
    "ndev", "age", "nuc", "aexp", "arexp", "asexp"
]

TARGET_COLUMN = "buggy"

EXCLUDED_COLUMNS = [
    "commit_id",
    "project",
    "fix",
    "year",
    "author_date"
]

EXPECTED_ROWS = 44834
EXPECTED_COLUMNS = 18


def main():
    print("\n=== ApacheJIT Dataset Validation ===\n")

    # 1. Check file
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Ask Person 1 to provide the processed CSV."
        )

    # 2. Load dataset
    df = pd.read_csv(DATA_PATH)

    print("Dataset loaded successfully!")
    print(f"Dataset shape: {df.shape}")

    # 3. Validate dimensions
    if len(df) != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS} rows, "
            f"but found {len(df)}."
        )

    if len(df.columns) != EXPECTED_COLUMNS:
        raise ValueError(
            f"Expected {EXPECTED_COLUMNS} columns, "
            f"but found {len(df.columns)}."
        )

    # 4. Verify required columns
    expected = set(
        FEATURE_COLUMNS
        + [TARGET_COLUMN]
        + EXCLUDED_COLUMNS
    )

    actual = set(df.columns)

    missing = expected - actual
    unexpected = actual - expected

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    if unexpected:
        raise ValueError(
            f"Unexpected columns: {sorted(unexpected)}"
        )

    # 5. Select ONLY approved model features
    X = df[FEATURE_COLUMNS].copy()

    # Convert features to numeric safely
    for column in FEATURE_COLUMNS:
        converted = pd.to_numeric(
            X[column], errors="coerce"
        )

        invalid = converted.isna() & X[column].notna()

        if invalid.any():
            raise ValueError(
                f"Non-numeric values found in {column}"
            )

        X[column] = converted

    # 6. Select target
    y = pd.to_numeric(
        df[TARGET_COLUMN], errors="coerce"
    )

    if y.isna().any():
        raise ValueError("Target contains missing or invalid values.")

    # 7. Check binary labels
    labels = set(y.unique())

    if not labels.issubset({0, 1}) or labels != {0, 1}:
        raise ValueError(
            f"Expected binary labels 0 and 1, got {labels}"
        )

    # 8. Report shapes
    print("\n=== Feature and Target Verification ===")

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    print(f"\nFeature count: {X.shape[1]}")
    print(f"Target column: {TARGET_COLUMN}")

    print("\nFeatures used:")
    for feature in X.columns:
        print(f" - {feature}")

    # 9. Missing-value analysis
    print("\n=== Missing Values ===")
    print(X.isnull().sum())

    print(
        f"\nTotal missing feature values: "
        f"{X.isnull().sum().sum()}"
    )

    print(f"Missing targets: {y.isnull().sum()}")

    # 10. Class distribution
    print("\n=== Target Distribution ===")
    print(y.value_counts().sort_index())

    print("\nClass percentages:")
    print(
        (y.value_counts(normalize=True) * 100)
        .sort_index()
        .round(2)
    )

    print("\n=== VALIDATION SUCCESSFUL ===")
    print("Dataset dimensions verified.")
    print("Only 12 approved features selected.")
    print("Metadata excluded from model inputs.")
    print("Target verified as binary.")


if __name__ == "__main__":
    main()
