import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

from api.main import app
from src import config

client = TestClient(app)

HIGH_RISK = {
    "age": 60, "gender": "male", "systolic_bp": 150, "diastolic_bp": 94,
    "glucose_fasting": 140, "hemoglobin": 12.8, "cholesterol_total": 250,
    "wbc_count": 8.0, "rbc_count": 4.5, "platelet_count": 200, "bmi": 32.0,
    "heart_rate": 90, "smoker": True, "alcohol_use": True,
    "family_history_diabetes": True, "family_history_heart": True,
    "fatigue": True, "shortness_of_breath": True, "chest_pain": True,
    "frequent_urination": True, "swelling": True, "jaundice": True, "dizziness": True,
}

LOW_RISK = {
    "age": 27, "gender": "female", "systolic_bp": 110, "diastolic_bp": 70,
    "glucose_fasting": 85, "hemoglobin": 13.9, "cholesterol_total": 165,
    "wbc_count": 6.2, "rbc_count": 4.8, "platelet_count": 275, "bmi": 21.0,
    "heart_rate": 65, "smoker": False, "alcohol_use": False,
    "family_history_diabetes": False, "family_history_heart": False,
    "fatigue": False, "shortness_of_breath": False, "chest_pain": False,
    "frequent_urination": False, "swelling": False, "jaundice": False, "dizziness": False,
}


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert set(resp.json()["models_available"]) == set(config.DISEASES)


def test_predict_all_returns_every_disease():
    resp = client.post("/predict/all", json=HIGH_RISK)
    assert resp.status_code == 200
    body = resp.json()["predictions"]
    assert set(body.keys()) == set(config.DISEASES)
    for disease, result in body.items():
        assert 0.0 <= result["probability"] <= 1.0
        assert result["risk_band"] in {"low", "elevated", "high"}


def test_predict_single_disease():
    resp = client.post("/predict/diabetes", json=HIGH_RISK)
    assert resp.status_code == 200
    assert resp.json()["disease"] == "diabetes"


def test_predict_unknown_disease_returns_404():
    resp = client.post("/predict/not_a_real_disease", json=HIGH_RISK)
    assert resp.status_code == 404


def test_predict_missing_field_returns_422():
    bad = dict(HIGH_RISK)
    del bad["age"]
    resp = client.post("/predict/diabetes", json=bad)
    assert resp.status_code == 422


def test_high_vs_low_risk_diverge():
    high = client.post("/predict/heart_disease", json=HIGH_RISK).json()
    low = client.post("/predict/heart_disease", json=LOW_RISK).json()
    assert high["probability"] > low["probability"]
