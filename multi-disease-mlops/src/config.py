from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

MLFLOW_TRACKING_URI = "file://" + str(ROOT_DIR / "mlruns")
MLFLOW_EXPERIMENT_NAME = "multi-disease-prediction"

RANDOM_STATE = 42
N_PATIENTS = 2200

# --- Shared patient-data schema ---
# Every disease model draws its inputs from this one pool, the way a real
# intake form or EHR collects it once per visit. Which subset each disease
# model actually uses is defined in DISEASE_FEATURES below - a diabetes
# model has no clinical reason to look at chest pain, for instance.
CATEGORICAL_COLS = ["gender"]
NUMERIC_COLS = [
    "age", "systolic_bp", "diastolic_bp", "glucose_fasting", "hemoglobin",
    "cholesterol_total", "wbc_count", "rbc_count", "platelet_count", "bmi",
    "heart_rate", "smoker", "alcohol_use", "family_history_diabetes",
    "family_history_heart", "fatigue", "shortness_of_breath", "chest_pain",
    "frequent_urination", "swelling", "jaundice", "dizziness",
]
ALL_FEATURE_COLS = CATEGORICAL_COLS + NUMERIC_COLS

# --- Per-disease target columns and the feature subset each one trains on ---
DISEASES = ["diabetes", "heart_disease", "anemia", "liver_disease", "kidney_disease", "hypertension"]

DISEASE_FEATURES = {
    "diabetes": {
        "categorical": [],
        "numeric": ["age", "bmi", "glucose_fasting", "family_history_diabetes", "frequent_urination", "fatigue"],
    },
    "heart_disease": {
        "categorical": ["gender"],
        "numeric": ["age", "systolic_bp", "cholesterol_total", "smoker", "family_history_heart",
                    "bmi", "heart_rate", "chest_pain", "shortness_of_breath"],
    },
    "anemia": {
        "categorical": ["gender"],
        "numeric": ["age", "hemoglobin", "rbc_count", "fatigue", "dizziness"],
    },
    "liver_disease": {
        "categorical": [],
        "numeric": ["age", "bmi", "alcohol_use", "jaundice", "fatigue", "platelet_count"],
    },
    "kidney_disease": {
        "categorical": [],
        "numeric": ["age", "systolic_bp", "glucose_fasting", "swelling",
                    "family_history_diabetes", "family_history_heart"],
    },
    "hypertension": {
        "categorical": [],
        "numeric": ["age", "systolic_bp", "diastolic_bp", "bmi", "smoker"],
    },
}

# --- Quality gate (same rationale as a single-disease clinical model: AUC
# in the 0.75-0.90 range on labs/symptoms data is realistic and expected;
# claiming near-1.0 AUC on this kind of data is a red flag, not a win) ---
MIN_ACCEPTABLE_AUC = 0.72
MIN_ACCEPTABLE_RECALL = 0.75
MIN_ACCEPTABLE_PRECISION = 0.28
