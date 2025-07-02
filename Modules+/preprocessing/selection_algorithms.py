# Modules 1.1/preprocessing/selection_algorithms.py

from typing import List, Dict, Any, Optional, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy


def normalize_features(
    X: pd.DataFrame,
    with_mean: bool = False,
    with_std: bool = True
) -> np.ndarray:
    """
    Scale features using StandardScaler.

    :param X: feature matrix (samples x features)
    :param with_mean: whether to center data
    :param with_std: whether to scale to unit variance
    :returns: scaled numpy array
    """
    scaler = StandardScaler(with_mean=with_mean, with_std=with_std)
    return scaler.fit_transform(X)


def lasso_feature_selection(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    cv_splits: int = 5,
    penalty: str = 'l1',
    solver: str = 'saga',
    scoring: str = 'roc_auc',
    max_iter: int = 5000
) -> List[str]:
    """
    Select features with non-zero coefficients from a L1-penalized logistic regression.

    :param X: features (DataFrame or array)
    :param y: binary labels
    :returns: list of column names selected
    """
    if isinstance(X, pd.DataFrame):
        cols = X.columns.tolist()
        X_vals = X.values
    else:
        X_vals = X
        cols = [f"f_{i}" for i in range(X_vals.shape[1])]

    lasso = LogisticRegressionCV(
        penalty=penalty,
        solver=solver,
        cv=StratifiedKFold(cv_splits),
        scoring=scoring,
        max_iter=max_iter
    )
    lasso.fit(X_vals, y)
    coefs = lasso.coef_[0]
    return [col for col, c in zip(cols, coefs) if c != 0]


def boruta_feature_selection(
    X: pd.DataFrame,
    y: Union[pd.Series, np.ndarray],
    estimator: Optional[Any] = None,
    max_iter: int = 100
) -> List[str]:
    """
    Use BorutaPy to identify all relevant features via a RandomForest.

    :param X: feature DataFrame
    :param y: target labels
    :param estimator: sklearn-compatible estimator (defaults to RF)
    :param max_iter: max Boruta iterations
    :returns: list of selected column names
    """
    if estimator is None:
        estimator = RandomForestClassifier(
            n_jobs=-1, class_weight='balanced', max_depth=5
        )

    boruta = BorutaPy(
        estimator,
        n_estimators='auto',
        max_iter=max_iter,
        random_state=0
    )
    boruta.fit(X.values, y.values)
    return X.columns[boruta.support_].tolist()


def stability_feature_selection(
    X: pd.DataFrame,
    y: Union[pd.Series, np.ndarray],
    n_resamples: int = 100,
    selection_threshold: float = 0.8,
    random_state: int = 42
) -> List[str]:
    """
    Stability selection via repeated L1-penalized logistic regression on bootstraps.

    :param X: feature DataFrame
    :param y: labels
    :param n_resamples: number of bootstrap iterations
    :param selection_threshold: fraction of times a feature must be selected
    :returns: stable feature list
    """
    counts = np.zeros(X.shape[1])
    for _ in range(n_resamples):
        # bootstrap sample
        idx = np.random.choice(X.index, size=len(X), replace=True)
        X_res = X.loc[idx].values
        y_res = y[idx] if hasattr(y, 'iloc') else y[idx]
        clf = LogisticRegression(penalty='l1', solver='saga', max_iter=5000)
        clf.fit(X_res, y_res)
        counts += (clf.coef_[0] != 0)

    selected = counts / n_resamples >= selection_threshold
    return X.columns[selected].tolist()


def combine_feature_sets(
    *feature_lists: List[str],
    min_agreement: int = 2
) -> List[str]:
    """
    From multiple feature lists, keep features selected by at least min_agreement methods.

    :param feature_lists: variable number of feature name lists
    :param min_agreement: minimum number of lists a feature must appear in
    :returns: combined feature list
    """
    from collections import Counter
    all_feats = Counter([f for fl in feature_lists for f in fl])
    return [f for f, cnt in all_feats.items() if cnt >= min_agreement]


def select_final_features(
    df: pd.DataFrame,
    label_col: str = 'type',
    methods: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    End-to-end feature selection pipeline combining methods.

    :param df: DataFrame with features and label
    :param label_col: name of the label column
    :param methods: dict of method names and their params, e.g. {
        'lasso': {...}, 'boruta': {...}, 'stability': {...}
    }
    :returns: DataFrame subset to final selected features + label
    """
    # split into X, y
    y = df[label_col]
    X = df.drop(columns=[label_col])

    # default methods if none provided
    if methods is None:
        methods = {
            'lasso': {},
            'boruta': {},
            # 'stability': {'n_resamples': 100, 'selection_threshold': 0.8}
        }

    # normalize once for lasso
    X_scaled = normalize_features(X)

    selected_sets = []
    if 'lasso' in methods:
        sel = lasso_feature_selection(X_scaled, y, **methods['lasso'])
        selected_sets.append(sel)
    if 'boruta' in methods:
        sel = boruta_feature_selection(X, y, **methods['boruta'])
        selected_sets.append(sel)
    if 'stability' in methods:
        sel = stability_feature_selection(X, y, **methods['stability'])
        selected_sets.append(sel)

    final_feats = combine_feature_sets(*selected_sets)
    # ensure at least 1 method overlap if too few
    if len(final_feats) < 2 and selected_sets:
        final_feats = combine_feature_sets(*selected_sets, min_agreement=1)

    # build DataFrame
    result = df[final_feats + [label_col]].copy()
    return result