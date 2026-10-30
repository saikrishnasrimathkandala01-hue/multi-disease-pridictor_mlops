"""
PSI-based drift monitor over the shared patient-data schema. Since every
disease model draws from the same input pool, one drift check covers all
six models - if the incoming patient population's labs/vitals shift from
the training distribution, every downstream model is affected.

Run:
    python monitoring/drift_check.py --incoming path/to/new_batch.csv
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src import config


def psi(baseline: pd.Series, incoming: pd.Series, bins: int = 10) -> float:
    breakpoints = np.quantile(baseline, np.linspace(0, 1, bins + 1))
    breakpoints[0], breakpoints[-1] = -np.inf, np.inf
    breakpoints = np.unique(breakpoints)

    base_counts = np.histogram(baseline, bins=breakpoints)[0] / len(baseline)
    inc_counts = np.histogram(incoming, bins=breakpoints)[0] / len(incoming)
    base_counts = np.clip(base_counts, 1e-6, None)
    inc_counts = np.clip(inc_counts, 1e-6, None)
    return float(np.sum((inc_counts - base_counts) * np.log(inc_counts / base_counts)))


CONTINUOUS_COLS = [
    "age", "systolic_bp", "diastolic_bp", "glucose_fasting", "hemoglobin",
    "cholesterol_total", "wbc_count", "rbc_count", "platelet_count", "bmi", "heart_rate",
]


def run_drift_check(incoming_path: str, psi_threshold: float = 0.25):
    baseline = pd.read_csv(config.DATA_DIR / "train.csv")
    incoming = pd.read_csv(incoming_path)

    report = []
    for col in CONTINUOUS_COLS:
        if col not in incoming.columns:
            continue
        score = psi(baseline[col], incoming[col])
        report.append({"feature": col, "psi": round(score, 4), "drifted": score > psi_threshold})

    report_df = pd.DataFrame(report).sort_values("psi", ascending=False)
    print(report_df.to_string(index=False))

    if report_df["drifted"].sum() > 0:
        print(f"\n⚠️  {report_df['drifted'].sum()} feature(s) exceeded PSI threshold {psi_threshold}. "
              "This affects every disease model, since they share this input pool - consider retraining all six.")
        sys.exit(1)
    print("\n✅ No significant drift detected across the shared patient-data schema.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--incoming", required=True)
    parser.add_argument("--threshold", type=float, default=0.25)
    args = parser.parse_args()
    run_drift_check(args.incoming, args.threshold)
