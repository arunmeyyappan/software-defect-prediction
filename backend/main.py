
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field


# ==========================================
# PROJECT CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "final_model.joblib"

FEATURE_COLUMNS = [
    "la", "ld", "nf", "nd", "ns", "ent",
    "ndev", "age", "nuc", "aexp", "arexp", "asexp"
]


# ==========================================
# LOAD TRAINED MODEL
# ==========================================

def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {MODEL_PATH}"
        )

    artifact = joblib.load(MODEL_PATH)

    if "pipeline" not in artifact:
        raise ValueError("Model artifact has no pipeline.")

    if artifact["feature_columns"] != FEATURE_COLUMNS:
        raise ValueError("Model feature schema mismatch.")

    return artifact


artifact = load_model()

model_pipeline = artifact["pipeline"]
model_name = artifact["model_name"]
threshold = float(artifact["threshold"])


# ==========================================
# FASTAPI APPLICATION
# ==========================================

app = FastAPI(
    title="ApacheJIT Software Defect Prediction API",
    description=(
        "Predict whether a software commit is likely "
        "to contain defects using machine learning."
    ),
    version="1.0.0"
)


# ==========================================
# CORS CONFIGURATION
# ==========================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501"
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# ==========================================
# REQUEST SCHEMA
# ==========================================

class PredictionRequest(BaseModel):

    model_config = ConfigDict(
        extra="forbid"
    )

    la: float = Field(allow_inf_nan=False)
    ld: float = Field(allow_inf_nan=False)
    nf: float = Field(allow_inf_nan=False)
    nd: float = Field(allow_inf_nan=False)
    ns: float = Field(allow_inf_nan=False)
    ent: float = Field(allow_inf_nan=False)

    ndev: float = Field(allow_inf_nan=False)
    age: float = Field(allow_inf_nan=False)
    nuc: float = Field(allow_inf_nan=False)
    aexp: float = Field(allow_inf_nan=False)
    arexp: float = Field(allow_inf_nan=False)
    asexp: float = Field(allow_inf_nan=False)


# ==========================================
# ROOT ENDPOINT
# ==========================================

@app.get("/")
def root():

    return {
        "message": "ApacheJIT Defect Prediction API",
        "status": "running",
        "documentation": "/docs"
    }


# ==========================================
# HEALTH ENDPOINT
# ==========================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": model_pipeline is not None,
        "model_name": model_name
    }


# ==========================================
# MODEL INFORMATION
# ==========================================

@app.get("/model-info")
def model_info():

    return {
        "model_name": model_name,
        "features_count": len(FEATURE_COLUMNS),
        "features": FEATURE_COLUMNS,
        "target": "buggy",
        "threshold": threshold,
        "training_rows": artifact["training_rows"]
    }


# ==========================================
# PREDICTION ENDPOINT
# ==========================================

@app.post("/predict")
def predict(request: PredictionRequest):

    try:

        # Convert the request to a DataFrame
        # while preserving the training feature order.
        input_data = pd.DataFrame(
            [
                {
                    feature: getattr(request, feature)
                    for feature in FEATURE_COLUMNS
                }
            ],
            columns=FEATURE_COLUMNS
        )

        probabilities = model_pipeline.predict_proba(
            input_data
        )[0]

        defective_probability = float(
            probabilities[1]
        )

        non_defective_probability = float(
            probabilities[0]
        )

        prediction = int(
            defective_probability >= threshold
        )

        label = (
            "Defective"
            if prediction == 1
            else "Non-defective"
        )

        return {
            "prediction": prediction,
            "label": label,
            "probability": {
                "non_defective": round(
                    non_defective_probability, 6
                ),
                "defective": round(
                    defective_probability, 6
                )
            },
            "threshold": threshold,
            "model_name": model_name
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}"
        ) from error
