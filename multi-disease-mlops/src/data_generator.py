"""
Synthetic patient dataset for the multi-disease prediction system.

Generates one shared pool of patient data - demographics, vitals, labs,
and symptoms, exactly the input types named in the project abstract - then
derives SIX separate binary disease labels (diabetes, heart_disease,
anemia, liver_disease, kidney_disease, hypertension), each from its own
clinically-motivated risk function over a relevant subset of that shared
data, plus irreducible noise.

No real patient data is used. Relationships are modeled after well-known
clinical risk factors (e.g. fasting glucose -> diabetes, hemoglobin ->
anemia, blood pressure -> hypertension) but are illustrative, not derived
from a real cohort. Noise is added deliberately so no disease is
near-perfectly separable - real lab/symptom-based prediction is noisy,
and a model claiming otherwise is more likely leaking information than
performing well.
"""
import numpy as np
import pandas as pd


def _sigmoid(z):
    return 1 / (1 + np.exp(-z))


def generate(n_patients: int = 1400, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # ---------- demographics ----------
    age = np.clip(rng.normal(50, 16, n_patients), 18, 90)
    gender = rng.choice(["male", "female"], size=n_patients, p=[0.5, 0.5])
    is_female = (gender == "female").astype(int)

    # ---------- lifestyle / history ----------
    smoker = rng.binomial(1, 0.22, n_patients)
    alcohol_use = rng.binomial(1, 0.25, n_patients)
    family_history_diabetes = rng.binomial(1, 0.22, n_patients)
    family_history_heart = rng.binomial(1, 0.22, n_patients)

    # ---------- vitals & labs ----------
    bmi = np.clip(rng.normal(26, 5, n_patients), 15, 50)
    systolic_bp = np.clip(rng.normal(122, 15, n_patients) + (age - 50) * 0.35 + (bmi - 26) * 0.5, 85, 200)
    diastolic_bp = np.clip(rng.normal(78, 10, n_patients) + (age - 50) * 0.12 + (bmi - 26) * 0.25, 50, 130)
    glucose_fasting = np.clip(rng.normal(95, 20, n_patients) + (age - 50) * 0.25 + (bmi - 26) * 0.9, 60, 300)
    hemoglobin = np.clip(rng.normal(14.2, 1.6, n_patients) - is_female * 1.2, 6, 18)
    cholesterol_total = np.clip(rng.normal(195, 35, n_patients) + (age - 50) * 0.4 + (bmi - 26) * 0.5, 100, 350)
    wbc_count = np.clip(rng.normal(7.5, 2.0, n_patients), 2, 20)
    rbc_count = np.clip(rng.normal(4.9, 0.55, n_patients) - is_female * 0.3, 2.5, 6.5)
    platelet_count = np.clip(rng.normal(250, 60, n_patients) - alcohol_use * 15, 100, 450)
    heart_rate = np.clip(rng.normal(75, 11, n_patients), 45, 130)

    # ---------- symptoms (correlated with relevant labs, not independent noise) ----------
    fatigue = rng.binomial(1, _sigmoid(-1.5 + 0.15 * (13 - hemoglobin) + 0.01 * (glucose_fasting - 100)
                                        + 0.02 * (age - 50)), n_patients)
    shortness_of_breath = rng.binomial(1, _sigmoid(-2.0 + 0.15 * (13 - hemoglobin) + 0.02 * (bmi - 26)
                                                     + 0.015 * (age - 50)), n_patients)
    chest_pain = rng.binomial(1, _sigmoid(-2.2 + 0.02 * (systolic_bp - 120) + 0.01 * (cholesterol_total - 200)
                                            + 0.35 * smoker + 0.015 * (age - 50)), n_patients)
    frequent_urination = rng.binomial(1, _sigmoid(-2.5 + 0.04 * (glucose_fasting - 100)
                                                    + 0.3 * family_history_diabetes), n_patients)
    swelling = rng.binomial(1, _sigmoid(-2.3 + 0.02 * (systolic_bp - 120) + 0.02 * (glucose_fasting - 100)
                                          + 0.01 * (age - 50)), n_patients)
    jaundice = rng.binomial(1, _sigmoid(-3.0 + 1.0 * alcohol_use + 0.01 * (bmi - 26)
                                          + 0.005 * (age - 50)), n_patients)
    dizziness = rng.binomial(1, _sigmoid(-2.0 + 0.15 * (13 - hemoglobin) + 0.01 * (age - 50)), n_patients)

    def sample_target(logit):
        noise = rng.normal(0, 1.0, n_patients)
        return rng.binomial(1, _sigmoid(logit + noise))

    diabetes = sample_target(
        -2.9 + 0.035 * age + 0.05 * (glucose_fasting - 100) + 0.07 * (bmi - 26)
        + 0.7 * family_history_diabetes + 0.5 * frequent_urination + 0.3 * fatigue
    )
    heart_disease = sample_target(
        -4.2 + 0.045 * age + 0.5 * (1 - is_female) + 0.022 * (systolic_bp - 120)
        + 0.016 * (cholesterol_total - 200) + 0.7 * smoker + 0.7 * family_history_heart
        + 0.035 * (bmi - 26) + 0.02 * (heart_rate - 75) + 1.1 * chest_pain + 0.6 * shortness_of_breath
    )
    anemia = sample_target(
        -1.6 + 0.9 * (13 - hemoglobin) + 0.6 * (5.0 - rbc_count) + 0.4 * fatigue
        + 0.4 * dizziness + 0.3 * is_female
    )
    liver_disease = sample_target(
        -3.6 + 0.03 * age + 0.08 * (bmi - 26) + 1.7 * alcohol_use + 2.0 * jaundice
        + 0.3 * fatigue - 0.016 * (platelet_count - 250)
    )
    kidney_disease = sample_target(
        -3.9 + 0.04 * age + 0.03 * (systolic_bp - 120) + 0.035 * (glucose_fasting - 100)
        + 1.3 * swelling + 0.6 * family_history_diabetes + 0.5 * family_history_heart
    )
    hypertension = sample_target(
        -3.8 + 0.05 * age + 0.075 * (systolic_bp - 120) + 0.06 * (diastolic_bp - 80)
        + 0.045 * (bmi - 26) + 0.4 * smoker
    )

    df = pd.DataFrame({
        "age": age.round(0).astype(int),
        "gender": gender,
        "systolic_bp": systolic_bp.round(0).astype(int),
        "diastolic_bp": diastolic_bp.round(0).astype(int),
        "glucose_fasting": glucose_fasting.round(0).astype(int),
        "hemoglobin": hemoglobin.round(1),
        "cholesterol_total": cholesterol_total.round(0).astype(int),
        "wbc_count": wbc_count.round(1),
        "rbc_count": rbc_count.round(2),
        "platelet_count": platelet_count.round(0).astype(int),
        "bmi": bmi.round(1),
        "heart_rate": heart_rate.round(0).astype(int),
        "smoker": smoker,
        "alcohol_use": alcohol_use,
        "family_history_diabetes": family_history_diabetes,
        "family_history_heart": family_history_heart,
        "fatigue": fatigue,
        "shortness_of_breath": shortness_of_breath,
        "chest_pain": chest_pain,
        "frequent_urination": frequent_urination,
        "swelling": swelling,
        "jaundice": jaundice,
        "dizziness": dizziness,
        "diabetes": diabetes,
        "heart_disease": heart_disease,
        "anemia": anemia,
        "liver_disease": liver_disease,
        "kidney_disease": kidney_disease,
        "hypertension": hypertension,
    })
    return df


if __name__ == "__main__":
    df = generate()
    for d in ["diabetes", "heart_disease", "anemia", "liver_disease", "kidney_disease", "hypertension"]:
        print(f"{d}: positive rate = {df[d].mean():.3f}")
