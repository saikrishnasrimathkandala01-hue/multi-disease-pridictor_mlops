"""
Serving layer for the Multi-Disease Prediction Platform.

Realizes the project's data-flow diagram:
    Patient Information -> Age + Blood Tests + Symptoms + Medical Reports
    -> Data Preprocessing -> ML Models -> Disease Prediction
    -> Prediction API (this file)

One shared PatientData payload is submitted once; each disease endpoint
(and /predict/all) selects the feature subset its own model was trained
on and returns an independent prediction - these are six separate models,
not one multi-label model (see README for why).

Run locally:
    uvicorn api.main:app --reload --port 8000
"""
import sys
import time
from pathlib import Path
from typing import Dict

sys.path.append(str(Path(__file__).resolve().parent.parent))

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src import config

app = FastAPI(
    title="Multi-Disease Prediction API",
    description="Predicts risk of diabetes, heart disease, anemia, liver disease, "
                "kidney disease, and hypertension from shared patient data.",
    version="1.0.0",
)

# Dev-convenience CORS: the frontend is served same-origin by this same app
# (see the StaticFiles mount at the bottom of this file), so CORS isn't
# actually needed for that path. This stays open in case you run the
# frontend separately (e.g. a different dev server) while prototyping -
# tighten this before any real deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_MODELS: Dict[str, dict] = {}


def get_model(disease: str) -> dict:
    if disease not in _MODELS:
        path = config.MODELS_DIR / f"{disease}.joblib"
        if not path.exists():
            raise HTTPException(status_code=503, detail=f"No trained model found for '{disease}'.")
        _MODELS[disease] = joblib.load(path)
    return _MODELS[disease]


class PatientData(BaseModel):
    """The shared 'Patient Information' payload - one submission feeds every disease model."""
    age: int = Field(..., ge=1, le=120)
    gender: str = Field(..., pattern="^(male|female)$")
    systolic_bp: float
    diastolic_bp: float
    glucose_fasting: float = Field(..., description="mg/dL")
    hemoglobin: float = Field(..., description="g/dL")
    cholesterol_total: float = Field(..., description="mg/dL")
    wbc_count: float = Field(..., description="10^3/µL")
    rbc_count: float = Field(..., description="10^6/µL")
    platelet_count: float = Field(..., description="10^3/µL")
    bmi: float
    heart_rate: float
    smoker: bool
    alcohol_use: bool
    family_history_diabetes: bool
    family_history_heart: bool
    fatigue: bool = False
    shortness_of_breath: bool = False
    chest_pain: bool = False
    frequent_urination: bool = False
    swelling: bool = False
    jaundice: bool = False
    dizziness: bool = False

    class Config:
        json_schema_extra = {
            "example": {
                "age": 58, "gender": "male", "systolic_bp": 148, "diastolic_bp": 92,
                "glucose_fasting": 138, "hemoglobin": 13.1, "cholesterol_total": 245,
                "wbc_count": 8.1, "rbc_count": 4.6, "platelet_count": 210, "bmi": 31.2,
                "heart_rate": 88, "smoker": True, "alcohol_use": False,
                "family_history_diabetes": True, "family_history_heart": True,
                "fatigue": True, "shortness_of_breath": True, "chest_pain": True,
                "frequent_urination": True, "swelling": False, "jaundice": False, "dizziness": False,
            }
        }


def _to_row(patient: PatientData) -> dict:
    return {
        "age": patient.age, "gender": patient.gender,
        "systolic_bp": patient.systolic_bp, "diastolic_bp": patient.diastolic_bp,
        "glucose_fasting": patient.glucose_fasting, "hemoglobin": patient.hemoglobin,
        "cholesterol_total": patient.cholesterol_total, "wbc_count": patient.wbc_count,
        "rbc_count": patient.rbc_count, "platelet_count": patient.platelet_count,
        "bmi": patient.bmi, "heart_rate": patient.heart_rate,
        "smoker": int(patient.smoker), "alcohol_use": int(patient.alcohol_use),
        "family_history_diabetes": int(patient.family_history_diabetes),
        "family_history_heart": int(patient.family_history_heart),
        "fatigue": int(patient.fatigue), "shortness_of_breath": int(patient.shortness_of_breath),
        "chest_pain": int(patient.chest_pain), "frequent_urination": int(patient.frequent_urination),
        "swelling": int(patient.swelling), "jaundice": int(patient.jaundice),
        "dizziness": int(patient.dizziness),
    }


def _predict_disease(disease: str, row: dict) -> dict:
    bundle = get_model(disease)
    X = pd.DataFrame([row])[bundle["feature_cols"]]
    proba = float(bundle["pipeline"].predict_proba(X)[0, 1])
    label = int(proba >= bundle["threshold"])
    band = "high" if proba >= 0.66 else ("elevated" if label else "low")
    return {
        "disease": disease, "prediction": label, "risk_band": band,
        "probability": round(proba, 4), "decision_threshold": bundle["threshold"],
    }


@app.get("/health")
def health():
    available = [d for d in config.DISEASES if (config.MODELS_DIR / f"{d}.joblib").exists()]
    return {"status": "ok", "models_available": available, "timestamp": time.time()}


@app.get("/model-info")
def model_info():
    info = {}
    for d in config.DISEASES:
        path = config.MODELS_DIR / f"{d}.joblib"
        if path.exists():
            bundle = joblib.load(path)
            info[d] = {
                "model_type": bundle["model_type"], "feature_names": bundle["feature_cols"],
                "decision_threshold": bundle["threshold"], "test_metrics": bundle["test_metrics"],
            }
    return info


@app.post("/predict/{disease}")
def predict_single(disease: str, patient: PatientData):
    if disease not in config.DISEASES:
        raise HTTPException(status_code=404, detail=f"Unknown disease '{disease}'. Choose from {config.DISEASES}.")
    row = _to_row(patient)
    result = _predict_disease(disease, row)
    result["note"] = "Screening estimate only - not a diagnosis. Consult a clinician for interpretation."
    return result


@app.post("/predict/all")
def predict_all(patient: PatientData):
    """Runs the same patient data through every trained disease model."""
    row = _to_row(patient)
    results = {d: _predict_disease(d, row) for d in config.DISEASES if (config.MODELS_DIR / f"{d}.joblib").exists()}
    return {
        "predictions": results,
        "note": "Screening estimates only - not a diagnosis. Consult a clinician for interpretation.",
    }


# Serve the frontend at "/" (same origin as the API, so no CORS needed in
# normal use). Mounted last so it never shadows the routes above -
# FastAPI/Starlette match explicit routes before falling through to a mount.
_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if _FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(_FRONTEND_DIR), html=True), name="frontend")
