# Modules 1.1/analysis/pruning.py

from typing import Tuple, List, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import roc_auc_score, roc_curve
from matplotlib.ticker import MaxNLocator
import joblib


def prune_misclassified(
    X: pd.DataFrame,
    y: pd.Series,
    mis_df: pd.DataFrame,
    min_mis: int = 50
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Drop samples misclassified at least `min_mis` times.

    :param X: feature DataFrame
    :param y: label Series
    :param mis_df: DataFrame with column 'n_misclassified' indexed by sample
    :param min_mis: threshold for pruning
    :returns: (X_clean, y_clean)
    """
    bad_idx = mis_df.loc[mis_df['n_misclassified'] >= min_mis].index
    X_clean = X.drop(index=bad_idx)
    y_clean = y.drop(index=bad_idx)
    return X_clean, y_clean


def rerun_repeated_cv(
    X: pd.DataFrame,
    y: pd.Series,
    C: float = 20,
    kernel: str = 'rbf',
    n_splits: int = 5,
    n_repeats: int = 100,
    random_state: int = 42
) -> Tuple[List[float], np.ndarray, pd.DataFrame]:
    """
    Repeated stratified CV on cleaned data, returning AUCs, interpolated TPRs grid, and per-sample stats.

    :returns:
      - aucs: list of per-fold AUCs
      - tprs: array of shape (n_folds, len(grid)) of interpolated TPRs
      - stats: DataFrame with 'n_misclassified' and 'sum_prob_wrong' for each sample
    """
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    grid = np.arange(0.0, 1.01, 0.01)

    aucs = []
    tprs = []
    stats = pd.DataFrame(
        0,
        index=X.index,
        columns=['n_misclassified', 'sum_prob_wrong']
    ).astype({'n_misclassified': int, 'sum_prob_wrong': float})

    clf_template = SVC(C=C, kernel=kernel, probability=True)

    for tr, te in cv.split(X, y):
        clf = clf_template.fit(X.iloc[tr], y.iloc[tr])
        probs = clf.predict_proba(X.iloc[te])[:, 1]
        aucs.append(roc_auc_score(y.iloc[te], probs))
        fpr, tpr, _ = roc_curve(y.iloc[te], probs)
        tprs.append(np.interp(grid, fpr, tpr))

        wrong = (probs >= 0.5).astype(int) != y.iloc[te].values
        idxs = X.index[te][wrong]
        stats.loc[idxs, 'n_misclassified'] += 1
        stats.loc[idxs, 'sum_prob_wrong'] += probs[wrong]

    return aucs, np.array(tprs), stats


def train_final_model(
    X: pd.DataFrame,
    y: pd.Series,
    C: float = 20,
    kernel: str = 'rbf'
) -> SVC:
    """
    Train a final SVC on the cleaned full dataset.

    :returns: fitted SVC model
    """
    clf = SVC(C=C, kernel=kernel, probability=True)
    return clf.fit(X, y)


def save_svm(
    model: SVC,
    path: str
) -> None:
    """
    Persist SVM model via joblib.
    """
    joblib.dump(model, path)


def select_kbest_curve(
    X: pd.DataFrame,
    y: pd.Series,
    k_list: List[int],
    C: float = 20,
    kernel: str = 'rbf',
    n_splits: int = 5,
    n_repeats: int = 10,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Evaluate cross-validated AUC for SelectKBest(mutual_info_classif, k).

    :returns: dict with 'k_list', 'mean_auc', 'std_auc', and 'best_k'
    """
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    scores_mean = []
    scores_std = []

    for k in k_list:
        pipe = SelectKBest(mutual_info_classif, k=k)
        X_sel = pipe.fit_transform(X, y)
        # wrap in SVC pipeline
        clf = SVC(C=C, kernel=kernel, probability=True, class_weight='balanced')
        scores = cross_val_score(clf, X_sel, y, cv=cv, scoring='roc_auc', n_jobs=-1)
        scores_mean.append(scores.mean())
        scores_std.append(scores.std())

    arr_mean = np.array(scores_mean)
    best_idx = arr_mean.argmax()
    return {
        'k_list': k_list,
        'mean_auc': scores_mean,
        'std_auc': scores_std,
        'best_k': k_list[best_idx]
    }


def plot_kbest_curve(
    results: Dict[str, Any],
    figsize: Tuple[int, int] = (8, 5)
) -> None:
    """
    Plot AUC vs. k with ±std band and highlight best_k.
    """
    k_list = results['k_list']
    mean_auc = results['mean_auc']
    std_auc = results['std_auc']
    best_k = results['best_k']

    plt.figure(figsize=figsize)
    plt.plot(k_list, mean_auc, marker='o')
    lower = np.array(mean_auc) - np.array(std_auc)
    upper = np.array(mean_auc) + np.array(std_auc)
    plt.fill_between(k_list, lower, upper, alpha=0.2)
    plt.axvline(best_k, ls='--', color='red')

    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel('Number of features (k)')
    plt.ylabel('Cross-validated AUC')
    plt.title('SelectKBest Learning Curve')
    plt.grid(True)
    plt.tight_layout()
    plt.show()