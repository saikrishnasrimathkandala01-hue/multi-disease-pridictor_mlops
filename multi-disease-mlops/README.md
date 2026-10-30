# Multi-Disease Prediction Platform (MLOps)

Predicts the likelihood of **six diseases** — diabetes, heart disease,
anemia, liver disease, kidney disease, and hypertension — from one shared
pool of patient data: demographics, vitals, blood-test results, and
reported symptoms.

```
Patient Information
   ↓
Age + Blood Tests + Symptoms + Medical Reports
   ↓
Data Preprocessing
   ↓
ML Models          (six separate models, one per disease)
   ↓
Disease Prediction
   ↓
Prediction API (FastAPI)
   ↓
MLflow Tracking
   ↓
Docker
   ↓
CI/CD
   ↓
Model Monitoring & Data Drift
```

> Educational / research prototype only, per the project's own scope —
> not a substitute for professional medical diagnosis. Every prediction
> is a screening estimate, not a diagnosis.

## Why six separate models instead of one

A single model cannot reliably predict arbitrary, clinically unrelated
diseases from the same inputs — diabetes, anemia, and liver disease have
almost nothing in common mechanistically, so forcing one model to learn
all of them at once either dilutes its accuracy on each, or requires a
properly multi-label-designed architecture beyond a college-project scope.

This project takes the more reliable path instead: **one shared input
schema, six independently trained binary classifiers**, each using only
the feature subset that's clinically relevant to it (see
`config.DISEASE_FEATURES`). A diabetes model has no reason to look at
chest pain; a liver-disease model has no reason to look at fasting blood
sugar. Submit patient data once through `/predict/all` and get six
independent, individually-validated predictions back.

## Shared patient-data schema

| Category | Fields |
|---|---|
| Demographics | age, gender |
| Vitals | systolic/diastolic blood pressure, heart rate, BMI |
| Labs | fasting glucose, hemoglobin, total cholesterol, WBC count, RBC count, platelet count |
| History | smoker, alcohol use, family history (diabetes / heart) |
| Symptoms | fatigue, shortness of breath, chest pain, frequent urination, swelling, jaundice, dizziness |

Each disease model draws from a clinically-motivated subset of this pool:

| Disease | Key inputs it actually uses |
|---|---|
| Diabetes | glucose, BMI, age, family history, frequent urination, fatigue |
| Heart disease | age, gender, blood pressure, cholesterol, smoking, family history, BMI, heart rate, chest pain, shortness of breath |
| Anemia | hemoglobin, RBC count, age, gender, fatigue, dizziness |
| Liver disease | alcohol use, jaundice, BMI, age, fatigue, platelet count |
| Kidney disease | blood pressure, glucose, swelling, family history, age |
| Hypertension | blood pressure, BMI, age, smoking |

## Realistic metrics, on purpose

All six models land at **test ROC AUC ≈ 0.77–0.83, recall ≈ 0.81–0.88** at
their tuned decision thresholds. That's a deliberate, realistic target —
symptom-and-labs clinical prediction is genuinely noisy; a model claiming
>0.95 AUC on this kind of data would be a red flag (leakage or
overfitting), not a good result. See `src/data_generator.py` for the
synthetic data's documented irreducible noise.

**Decision thresholds are tuned for recall, not accuracy.** Each model's
threshold is chosen as the strictest cutoff that still catches at least
~84% of true cases on the validation set (`src/train.py::tune_threshold`)
— in a screening context, missing a true case is costlier than a false
alarm that leads to a follow-up test. This means precision runs lower
(~0.35–0.60) than a naive accuracy-optimized model would show; that's the
intended tradeoff, not an oversight.

## Two ways to try the prototype

| | `frontend/` | `standalone-website/` |
|---|---|---|
| Requires | Backend running (`uvicorn`) | Nothing — just open the file |
| Predictions computed by | Real FastAPI backend, live | JavaScript, using exported model weights |
| Use case | Realistic client/server demo | Share a link, no setup, offline-friendly |

Both render the same design and hit the same 6 models; `frontend/` is
served by the backend itself at `http://localhost:8000/` (see below),
while `standalone-website/index.html` can be opened directly or hosted
anywhere as a static file.

## Getting started

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python data/prepare_data.py     # generates synthetic patients, splits data
python src/train.py               # trains & gates all 6 disease models
python src/train.py --disease diabetes   # retrain just one
python src/evaluate.py            # CI-style re-validation of every model

uvicorn api.main:app --reload --port 8000
```

### Example: predict all six diseases from one patient submission

```bash
curl -X POST http://localhost:8000/predict/all -H "Content-Type: application/json" -d '{
  "age": 58, "gender": "male", "systolic_bp": 148, "diastolic_bp": 92,
  "glucose_fasting": 138, "hemoglobin": 13.1, "cholesterol_total": 245,
  "wbc_count": 8.1, "rbc_count": 4.6, "platelet_count": 210, "bmi": 31.2,
  "heart_rate": 88, "smoker": true, "alcohol_use": false,
  "family_history_diabetes": true, "family_history_heart": true,
  "fatigue": true, "shortness_of_breath": true, "chest_pain": true,
  "frequent_urination": true, "swelling": false, "jaundice": false, "dizziness": false
}'
```

Or predict just one: `POST /predict/diabetes` with the same payload shape.

### Docker / tests / drift monitoring

```bash
docker compose up --build         # API on :8000, MLflow UI on :5000
pytest tests/ -v
python monitoring/drift_check.py --incoming data/processed/test.csv
```

One drift check covers all six models, since they share the same input
pool — if the incoming patient population shifts, every downstream model
is potentially affected.

## Project layout

```
src/config.py            Shared schema, per-disease feature subsets, quality-gate thresholds
src/data_generator.py     Synthetic multi-disease patient data (or swap in a real dataset)
data/prepare_data.py       Generates + splits data (stratified on diagnosis count)
src/train.py                Trains & gates all 6 models, MLflow tracking, threshold tuning
src/evaluate.py              CI re-validation across all 6 saved artifacts
api/main.py                   FastAPI: /predict/{disease}, /predict/all, /health, /model-info
tests/                          API + model-quality tests, parametrized across all 6 diseases
monitoring/drift_check.py        PSI drift monitor over the shared schema
Dockerfile / docker-compose.yml   Containerized API + local MLflow server
.github/workflows/ci-cd.yml        CI/CD: prep → train (all 6) → gate → test → build/push → deploy
frontend/index.html                 Frontend served by the backend, calls /predict/all live
standalone-website/index.html        No-backend static version (model weights embedded)
models/*.joblib                       Pre-trained artifacts, included so it runs immediately
```

## Pushing this to GitHub

The repo is already structured for it — `.gitignore` excludes
`__pycache__`, `mlruns/`, `data/raw|processed`, and regeneratable model
files, and `.github/workflows/ci-cd.yml` is a ready-to-run pipeline
(installs deps, regenerates data, retrains and gates all six models,
re-validates, runs tests, builds/pushes a Docker image on `main`).

```bash
cd multi-disease-mlops
git init
git add .
git commit -m "Initial commit: multi-disease MLOps platform"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

Once pushed, the Actions tab will run the CI/CD workflow automatically.
A few things worth doing before/after that first push:
- The included `models/*.joblib` files are committed as-is if you don't
  remove them from `.gitignore`'s exclusion — by default `.gitignore`
  excludes `models/*.joblib`, so the CI workflow's own `train.py` step is
  what (re)produces them; the copies in this zip are for immediately
  running the demo locally without training first.
- If you enable the `build-and-push-image` job, `GITHUB_TOKEN` is
  provided automatically by Actions for `ghcr.io` — no extra secret setup
  needed for that step.
- `docker-compose.yml`'s `mlflow` service and `data/`, `mlruns/` are
  local-dev conveniences; a real deployment would point `MLFLOW_TRACKING_URI`
  at a shared tracking server instead.

## Limitations

- **Synthetic data.** No real patient records are used. Relationships
  (e.g. glucose → diabetes) are modeled after well-known clinical risk
  factors but are illustrative, not fit to a real cohort. Swap
  `src/data_generator.py::generate()` for a real multi-disease dataset
  loader to move past prototype stage.
- **Six independent models, not a joint multi-label model.** A patient
  with correlated conditions (e.g. diabetes + kidney disease, a common
  real comorbidity) is scored by each model independently; the system
  doesn't currently model cross-disease correlation explicitly.
- **Educational/research scope**, per the project abstract — not
  validated for clinical use.
