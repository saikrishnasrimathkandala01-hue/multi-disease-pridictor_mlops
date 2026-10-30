import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import joblib
import pandas as pd
import pytest
from sklearn.metrics import recall_score, roc_auc_score

from src import config


@pytest.mark.parametrize("disease", config.DISEASES)
def test_model_artifact_exists(disease):
    assert (config.MODELS_DIR / f"{disease}.joblib").exists(), f"Run `python src/train.py` for {disease}."


@pytest.mark.parametrize("disease", config.DISEASES)
def test_model_meets_quality_gate(disease):
    bundle = joblib.load(config.MODELS_DIR / f"{disease}.joblib")
    pipeline, threshold = bundle["pipeline"], bundle["threshold"]

    test_df = pd.read_csv(config.DATA_DIR / "test.csv")
    X_test, y_test = test_df[bundle["feature_cols"]], test_df[disease]

    proba = pipeline.predict_proba(X_test)[:, 1]
    preds = (proba >= threshold).astype(int)

    assert roc_auc_score(y_test, proba) >= config.MIN_ACCEPTABLE_AUC
    assert recall_score(y_test, preds) >= config.MIN_ACCEPTABLE_RECALL
