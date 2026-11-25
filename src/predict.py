
from pathlib import Path
import argparse
import json

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "final_model.joblib"

FEATURE_COLUMNS = [
    "la", "ld", "nf", "nd", "ns", "ent",
    "ndev", "age", "nuc", "aexp", "arexp", "asexp"
]


def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    artifact = joblib.load(MODEL_PATH)

    if artifact["feature_columns"] != FEATURE_COLUMNS:
        raise ValueError("Feature schema mismatch.")

    return artifact


def validate_input(features):

    if not isinstance(features, dict):
        raise TypeError("Input must be a dictionary.")

    missing = set(FEATURE_COLUMNS) - set(features)
    extra = set(features) - set(FEATURE_COLUMNS)

    if missing:
        raise ValueError(
            f"Missing features: {sorted(missing)}"
        )

    if extra:
        raise ValueError(
            f"Unexpected features: {sorted(extra)}"
        )

    try:
        values = [
            float(features[column])
            for column in FEATURE_COLUMNS
        ]
    except (ValueError, TypeError) as error:
        raise ValueError(
            "Every feature must have a numeric value."
        ) from error

    if not np.isfinite(values).all():
        raise ValueError(
            "Features must contain finite numbers."
        )

    return pd.DataFrame(
        [values],
        columns=FEATURE_COLUMNS
    )


def predict_one(features, artifact=None):

    if artifact is None:
        artifact = load_model()

    model = artifact["pipeline"]
    threshold = float(artifact["threshold"])

    X = validate_input(features)

    probabilities = model.predict_proba(X)[0]

    defective_probability = float(probabilities[1])
    non_defective_probability = float(probabilities[0])

    prediction = int(
        defective_probability >= threshold
    )

    return {
        "prediction": prediction,
        "label": (
            "Defective"
            if prediction == 1
            else "Non-defective"
        ),
        "probability": {
            "non_defective": round(
                non_defective_probability, 6
            ),
            "defective": round(
                defective_probability, 6
            )
        },
        "threshold": threshold,
        "model_name": artifact["model_name"]
    }


def predict_batch(input_path, output_path):

    artifact = load_model()

    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    df = pd.read_csv(input_path)

    missing = set(FEATURE_COLUMNS) - set(df.columns)
    extra = set(df.columns) - set(FEATURE_COLUMNS)

    if missing or extra:
        raise ValueError(
            f"Invalid CSV schema. "
            f"Missing: {sorted(missing)}, "
            f"Extra: {sorted(extra)}"
        )

    X = df[FEATURE_COLUMNS].apply(
        pd.to_numeric,
        errors="raise"
    ).astype(float)

    values = X.to_numpy()

    if not np.isfinite(values).all():
        raise ValueError(
            "Input CSV contains missing or non-finite values."
        )

    model = artifact["pipeline"]

    probabilities = model.predict_proba(X)[:, 1]

    threshold = float(artifact["threshold"])

    predictions = (
        probabilities >= threshold
    ).astype(int)

    output = df.copy()

    output["prediction"] = predictions

    output["prediction_label"] = [
        "Defective" if value == 1 else "Non-defective"
        for value in predictions
    ]

    output["defective_probability"] = probabilities

    output["model_name"] = artifact["model_name"]

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output.to_csv(output_path, index=False)

    print(f"Predictions saved to: {output_path}")
    print(f"Total records predicted: {len(output)}")



def main():

    parser = argparse.ArgumentParser(
        description="ApacheJIT defect prediction"
    )

    parser.add_argument(
        "--input-json",
        help="Single software commit feature JSON"
    )

    parser.add_argument(
        "--input-json-file",
        help="Path to a JSON input file"
    )

    parser.add_argument(
        "--input-csv",
        help="CSV containing exactly the 12 model features"
    )

    parser.add_argument(
        "--output-csv",
        default="reports/batch_predictions.csv"
    )

    args = parser.parse_args()

    # Exactly one input method must be supplied.
    input_count = sum([
        args.input_json is not None,
        args.input_json_file is not None,
        args.input_csv is not None
    ])

    if input_count != 1:
        parser.error(
            "Provide exactly one input: "
            "--input-json, --input-json-file, or --input-csv"
        )

    if args.input_json is not None:

        features = json.loads(args.input_json)

        result = predict_one(features)

        print(json.dumps(result, indent=4))

    elif args.input_json_file is not None:

        input_path = Path(args.input_json_file)

        if not input_path.is_absolute():
            input_path = ROOT / input_path

        with input_path.open(
            "r",
            encoding="utf-8-sig"
        ) as file:

            features = json.load(file)

        result = predict_one(features)

        print(json.dumps(result, indent=4))

    else:

        predict_batch(
            args.input_csv,
            args.output_csv
        )


if __name__ == "__main__":
    main()