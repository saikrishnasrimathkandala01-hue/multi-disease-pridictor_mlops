"""
Generates the synthetic multi-disease patient dataset and splits it into
train/val/test. Stratifies on the count of positive diagnoses per patient
(0 through 6) rather than any single disease, so all six labels stay
reasonably balanced across the splits at once.

Run:
    python data/prepare_data.py
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from sklearn.model_selection import train_test_split

from src import config, data_generator

RAW_DIR = Path(__file__).parent / "raw"
PROCESSED_DIR = Path(__file__).parent / "processed"


def prepare(val_frac: float = 0.15, test_frac: float = 0.15):
    RAW_DIR.mkdir(exist_ok=True)
    PROCESSED_DIR.mkdir(exist_ok=True)

    df = data_generator.generate(n_patients=config.N_PATIENTS, seed=config.RANDOM_STATE)
    df.to_csv(RAW_DIR / "patients.csv", index=False)

    strat_key = df[config.DISEASES].sum(axis=1).clip(upper=3)  # bucket 3+ together to avoid tiny classes

    train_df, temp_df, strat_train, strat_temp = train_test_split(
        df, strat_key, test_size=(val_frac + test_frac), stratify=strat_key, random_state=config.RANDOM_STATE
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=test_frac / (val_frac + test_frac),
        stratify=strat_temp, random_state=config.RANDOM_STATE,
    )

    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    val_df.to_csv(PROCESSED_DIR / "val.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)

    print(f"train={len(train_df)} val={len(val_df)} test={len(test_df)}")
    for d in config.DISEASES:
        print(f"  {d:<15} pos rate: train={train_df[d].mean():.3f} "
              f"val={val_df[d].mean():.3f} test={test_df[d].mean():.3f}")
    return train_df, val_df, test_df


if __name__ == "__main__":
    prepare()
