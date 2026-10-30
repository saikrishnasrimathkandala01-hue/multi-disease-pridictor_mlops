"""
Re-validates every saved disease model artifact against its held-out test
set. Used as a CI gate - exits non-zero if any disease's artifact fails,
blocking the build/deploy job.

Run:
    python src/evaluate.py
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import joblib
import pandas as pd
from sklearn.metrics import recall_score, roc_auc_score

from src import config


def evaluate_disease(disease: str) -> bool:
    model_path = config.MODELS_DIR / f"{disease}.joblib"
    if not model_path.exists():
        print(f"[{disease}] SKIP - no saved artifact (training likely failed its gate)")
        return False

    bundle = joblib.load(model_path)
    pipeline, threshold = bundle["pipeline"], bundle["threshold"]

    test_df = pd.read_csv(config.DATA_DIR / "test.csv")
    X_test, y_test = test_df[bundle["feature_cols"]], test_df[disease]

    proba = pipeline.predict_proba(X_test)[:, 1]
    preds = (proba >= threshold).astype(int)
    auc = roc_auc_score(y_test, proba)
    recall = recall_score(y_test, preds)

    passed = auc >= config.MIN_ACCEPTABLE_AUC and recall >= config.MIN_ACCEPTABLE_RECALL
    print(f"[{disease}] test_auc={auc:.4f} test_recall={recall:.4f} -> {'PASS' if passed else 'FAIL'}")
    return passed


def main():
    results = {d: evaluate_disease(d) for d in config.DISEASES}
    if not all(results.values()):
        print("\nFAIL: one or more disease models did not meet the deployment quality gate.")
        sys.exit(1)
    print("\nPASS: all disease models meet the deployment quality gate.")


if __name__ == "__main__":
    main()
