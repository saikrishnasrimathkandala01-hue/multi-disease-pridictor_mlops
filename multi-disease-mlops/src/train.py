"""
Trains one binary classifier per disease (diabetes, heart_disease, anemia,
liver_disease, kidney_disease, hypertension), each using only the feature
subset clinically relevant to it (config.DISEASE_FEATURES). Every
candidate/disease combination is logged to MLflow; the best candidate per
disease is selected by validation ROC AUC, its decision threshold is tuned
for recall (screening priority - see config.py), and it's gated before
being saved.

Run:
    python src/train.py
    python src/train.py --disease diabetes   # retrain just one
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config


def load_split(name: str) -> pd.DataFrame:
    return pd.read_csv(config.DATA_DIR / f"{name}.csv")


def build_preprocessor(numeric_cols, categorical_cols):
    transformers = [("num", StandardScaler(), numeric_cols)]
    if categorical_cols:
        transformers.append(("cat", OneHotEncoder(drop="if_binary"), categorical_cols))
    return ColumnTransformer(transformers)


def build_candidates():
    return {
        "logistic_regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=config.RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=250, max_depth=5, class_weight="balanced", random_state=config.RANDOM_STATE
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=120, max_depth=2, random_state=config.RANDOM_STATE
        ),
    }


def tune_threshold(y_true, proba, target_recall: float = 0.80):
    """
    Pick the STRICTEST threshold (best precision) that still achieves at
    least `target_recall` - i.e. "the most selective cutoff that still
    catches at least 80% of true cases." This is what a recall-priority
    screening tool actually wants: not the lowest threshold that trivially
    flags everyone (which maximizes recall uselessly), and not a pure
    sensitivity/specificity balance point (Youden's J), which can undershoot
    the recall the quality gate requires.
    """
    candidates = []
    for t in np.arange(0.95, 0.04, -0.01):
        preds = (proba >= t).astype(int)
        if preds.sum() == 0:
            continue
        rec = recall_score(y_true, preds)
        if rec >= target_recall:
            prec = precision_score(y_true, preds, zero_division=0)
            candidates.append((t, prec, rec))
            break  # thresholds are descending, so the first hit is the strictest

    if candidates:
        t, prec, rec = candidates[0]
        return float(t), {"precision": prec, "recall": rec}

    # target recall unreachable at any threshold - fall back to the lowest
    # threshold tested (maximum achievable recall for this model)
    preds = (proba >= 0.05).astype(int)
    return 0.05, {
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds),
    }


def train_one_disease(disease: str, train_df, val_df, test_df) -> bool:
    feats = config.DISEASE_FEATURES[disease]
    num_cols, cat_cols = feats["numeric"], feats["categorical"]
    feature_cols = cat_cols + num_cols

    X_train, y_train = train_df[feature_cols], train_df[disease]
    X_val, y_val = val_df[feature_cols], val_df[disease]
    X_test, y_test = test_df[feature_cols], test_df[disease]

    best = {"name": None, "pipeline": None, "auc": -1, "threshold": 0.5, "val_metrics": None}

    for name, estimator in build_candidates().items():
        with mlflow.start_run(run_name=f"{disease}__{name}"):
            mlflow.set_tag("disease", disease)
            pipeline = Pipeline([("pre", build_preprocessor(num_cols, cat_cols)), ("clf", estimator)])
            pipeline.fit(X_train, y_train)

            val_proba = pipeline.predict_proba(X_val)[:, 1]
            val_auc = roc_auc_score(y_val, val_proba)
            threshold, thresh_metrics = tune_threshold(y_val, val_proba, target_recall=config.MIN_ACCEPTABLE_RECALL + 0.09)

            mlflow.log_params({"model_type": name, "disease": disease, **estimator.get_params()})
            mlflow.log_metrics({
                "val_auc": val_auc, "val_recall_at_threshold": thresh_metrics["recall"],
                "val_precision_at_threshold": thresh_metrics["precision"], "chosen_threshold": threshold,
            })
            mlflow.sklearn.log_model(pipeline, artifact_path="model")

            print(f"  [{disease}/{name}] val_auc={val_auc:.3f} threshold={threshold:.2f} "
                  f"recall={thresh_metrics['recall']:.3f} precision={thresh_metrics['precision']:.3f}")

            if val_auc > best["auc"]:
                best.update(name=name, pipeline=pipeline, auc=val_auc, threshold=threshold, val_metrics=thresh_metrics)

    test_proba = best["pipeline"].predict_proba(X_test)[:, 1]
    test_preds = (test_proba >= best["threshold"]).astype(int)
    test_metrics = {
        "auc": roc_auc_score(y_test, test_proba),
        "recall": recall_score(y_test, test_preds),
        "precision": precision_score(y_test, test_preds, zero_division=0),
        "accuracy": accuracy_score(y_test, test_preds),
    }

    passed_gate = best["auc"] >= config.MIN_ACCEPTABLE_AUC and best["val_metrics"]["recall"] >= config.MIN_ACCEPTABLE_RECALL

    print(f"  -> best={best['name']} val_auc={best['auc']:.3f} | test={test_metrics} | gate={'PASS' if passed_gate else 'FAIL'}")

    if passed_gate:
        joblib.dump(
            {
                "disease": disease,
                "pipeline": best["pipeline"],
                "feature_cols": feature_cols,
                "categorical_cols": cat_cols,
                "numeric_cols": num_cols,
                "threshold": float(best["threshold"]),
                "test_metrics": test_metrics,
                "model_type": best["name"],
            },
            config.MODELS_DIR / f"{disease}.joblib",
        )
    return passed_gate


def main(only_disease: str = None):
    train_df, val_df, test_df = load_split("train"), load_split("val"), load_split("test")

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)

    diseases = [only_disease] if only_disease else config.DISEASES
    results = {}
    for disease in diseases:
        print(f"\n=== Training: {disease} ===")
        results[disease] = train_one_disease(disease, train_df, val_df, test_df)

    print("\n=== Summary ===")
    for disease, passed in results.items():
        print(f"  {disease:<15} {'PASSED - saved' if passed else 'FAILED - not deployed'}")

    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--disease", choices=config.DISEASES, default=None)
    args = parser.parse_args()
    main(args.disease)
