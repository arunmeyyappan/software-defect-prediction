
from pathlib import Path

import joblib
import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "final_model.joblib"

FEATURES = [
    "la", "ld", "nf", "nd", "ns", "ent",
    "ndev", "age", "nuc", "aexp", "arexp", "asexp"
]

SAMPLE = {
    "la": 20,
    "ld": 10,
    "nf": 5,
    "nd": 3,
    "ns": 2,
    "ent": 1.5,
    "ndev": 4,
    "age": 30,
    "nuc": 3,
    "aexp": 2,
    "arexp": 1,
    "asexp": 2
}


def load_artifact():
    assert MODEL_PATH.exists(), "Model artifact is missing."
    return joblib.load(MODEL_PATH)


def test_model_artifact_exists():
    artifact = load_artifact()
    assert "pipeline" in artifact
    assert "model_name" in artifact


def test_model_uses_12_features():
    artifact = load_artifact()
    assert artifact["feature_columns"] == FEATURES


def test_model_prediction_shape():
    artifact = load_artifact()

    model = artifact["pipeline"]

    X = pd.DataFrame([SAMPLE], columns=FEATURES)

    prediction = model.predict(X)

    assert prediction.shape == (1,)


def test_model_probability_output():
    artifact = load_artifact()

    model = artifact["pipeline"]

    X = pd.DataFrame([SAMPLE], columns=FEATURES)

    probabilities = model.predict_proba(X)

    assert probabilities.shape == (1, 2)

    assert np.isfinite(probabilities).all()

    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)

    assert np.allclose(
        probabilities.sum(axis=1),
        1.0
    )


def test_prediction_is_binary():
    artifact = load_artifact()

    model = artifact["pipeline"]

    X = pd.DataFrame([SAMPLE], columns=FEATURES)

    prediction = model.predict(X)[0]

    assert prediction in (0, 1)
