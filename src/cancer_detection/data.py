"""Cleaning, balancing and splitting the RSNA metadata (train.csv)."""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

DENSITY = {"A": 0, "B": 1, "C": 2, "D": 3}


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows without age or density (no imputation on medical data) and the unused columns."""
    out = df.dropna(subset=["age", "density"]).drop(columns=["site_id", "implant"], errors="ignore")
    return out.assign(density=out["density"].map(DENSITY), age=out["age"].astype(int))


def balance_by_image(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Every positive image plus as many distinct negative images, drawn at random.

    The notebook drew negatives with replacement (np.random.choice's default), so some were
    picked twice and the set ended up 664/656 instead of an exact 50/50.
    """
    pos = df[df["cancer"] == 1]
    neg = df[df["cancer"] == 0].sample(n=len(pos), random_state=seed)
    return pd.concat([pos, neg]).sort_index()


def balance_by_patient(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Every image of every positive patient plus the images of as many negative patients."""
    has_cancer = df.groupby("patient_id")["cancer"].max()
    pos = has_cancer.index[has_cancer == 1]
    neg = has_cancer.index[has_cancer == 0]
    rng = np.random.default_rng(seed)
    chosen = np.concatenate([pos, rng.choice(neg, size=min(len(pos), len(neg)), replace=False)])
    return df[df["patient_id"].isin(chosen)]


def patient_split(df: pd.DataFrame, test_size: float, seed: int = 0):
    """Train/test split that keeps every image of a patient on the same side.

    The notebook split by image, so the two views of one breast could land in train and
    validation at once and inflate validation scores.
    """
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(df, groups=df["patient_id"]))
    return df.iloc[train_idx], df.iloc[test_idx]
