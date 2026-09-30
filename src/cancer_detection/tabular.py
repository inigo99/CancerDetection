"""The tabular baseline: an SVM on age and density."""

import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

FEATURES = ["age", "density"]


def svm_baseline(normalise: bool = False):
    """The thesis's final tabular model. The scaler lives in the pipeline, so it is fitted on
    the training data only (the notebook fitted a separate scaler on each split)."""
    svm = SVC(gamma="auto", decision_function_shape="ovo", random_state=0)
    return make_pipeline(StandardScaler(), svm) if normalise else svm


def fit(train: pd.DataFrame, normalise: bool = False):
    return svm_baseline(normalise).fit(train[FEATURES], train["cancer"])
