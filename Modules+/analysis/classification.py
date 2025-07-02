# Modules 1.1/analysis/classification.py

from typing import Tuple, Dict, Any, List, Union, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, RepeatedStratifiedKFold
)
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, confusion_matrix, roc_auc_score


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Stratified train/test split.

    :returns: X_train, X_test, y_train, y_test
    """
    return train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )


def build_svm_pipeline(
    C: float = 20,
    kernel: str = "rbf",
    probability: bool = True,
    class_weight: Union[str, Dict[Any, Any]] = 'balanced'
) -> Pipeline:
    """
    Create an SVM pipeline with scaling.

    :returns: sklearn Pipeline
    """
    return Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(C=C, kernel=kernel, probability=probability, class_weight=class_weight))
    ])


def run_cross_validated_model_with_counts(
    X: pd.DataFrame,
    y: pd.Series,
    C: float = 20,
    kernel: str = 'rbf',
    cv_splits: int = 5,
    ci: float = 0.99,
    random_state: int = 42,
    plot: bool = True
) -> Tuple[Dict[str, Any], List[np.ndarray]]:
    """
    5-fold stratified CV of SVM, tracking metrics and misclassification counts.

    :returns: (results_dict, list_of_confusion_matrices)
    """
    kf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    mean_fpr = np.linspace(0, 1, 100)

    folds_auc: List[float] = []
    folds_conf: List[np.ndarray] = []
    tprs_interp: List[np.ndarray] = []
    mis_count = pd.Series(0, index=X.index, name='Times_Misclassified')

    if plot:
        plt.figure(figsize=(10, 8))

    for fold, (train_idx, test_idx) in enumerate(kf.split(X, y), start=1):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        pipe = build_svm_pipeline(C=C, kernel=kernel)
        pipe.fit(X_tr, y_tr)

        probs = pipe.predict_proba(X_te)[:, 1]
        preds = pipe.predict(X_te)

        fpr, tpr, _ = roc_curve(y_te, probs)
        roc_auc = auc(fpr, tpr)
        folds_auc.append(roc_auc)
        interp = np.interp(mean_fpr, fpr, tpr)
        interp[0] = 0.0
        tprs_interp.append(interp)
        folds_conf.append(confusion_matrix(y_te, preds))

        mis_mask = preds != y_te
        mis_count.loc[X_te.index[mis_mask]] += 1

        if plot:
            plt.plot(fpr, tpr, lw=1.5, alpha=0.6,
                     label=f'Fold {fold} (AUC={roc_auc:.2f})')

    # compute aggregated metrics
    tprs_arr = np.array(tprs_interp)
    mean_tpr = tprs_arr.mean(axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)

    # AUC confidence interval from fold AUCs
    auc_std = np.std(folds_auc)
    z = np.abs(np.percentile(folds_auc, [(1-ci)/2*100, (1+ci)/2*100]))
    auc_ci = (np.mean(folds_auc) - z[1]*auc_std/np.sqrt(len(folds_auc)),
              np.mean(folds_auc) + z[1]*auc_std/np.sqrt(len(folds_auc)))

    if plot:
        lower = np.percentile(tprs_arr, ((1-ci)/2)*100, axis=0)
        upper = np.percentile(tprs_arr, (1-(1-ci)/2)*100, axis=0)
        plt.fill_between(mean_fpr, lower, upper, color='grey', alpha=0.3,
                         label=f'{int(ci*100)}% CI')
        plt.plot([0,1],[0,1],'k--', lw=1.5)
        plt.xlabel('FPR'); plt.ylabel('TPR')
        plt.title('5-Fold ROC Curves')
        plt.legend(loc='lower right'); plt.tight_layout(); plt.show()

    # misclassification summary
    mis_df = mis_count.reset_index().rename(columns={'index':'Index'})
    mis_df = mis_df[mis_df['Times_Misclassified']>0].sort_values('Times_Misclassified', ascending=False)

    results = {
        'mean_fpr': mean_fpr,
        'mean_tpr': mean_tpr,
        'mean_auc': mean_auc,
        'auc_ci': auc_ci,
        'fold_aucs': folds_auc,
        'confusion_matrices': folds_conf,
        'misclassification_counts': mis_df
    }
    return results, folds_conf


def plot_mean_roc_with_deciles(
    mean_fpr: np.ndarray,
    mean_tpr: np.ndarray,
    mean_auc: float,
    n_deciles: int = 10
) -> pd.DataFrame:
    """
    Plot mean ROC and decile trade-offs.

    :returns: DataFrame of FPR ranges and avg TPR
    """
    import numpy as _np
    plt.figure(figsize=(10, 8))
    plt.plot(mean_fpr, mean_tpr, lw=2.5,
             label=f'Mean ROC (AUC={mean_auc:.2f})')
    plt.plot([0,1],[0,1],'k--', lw=1.5)

    bins = _np.linspace(0,1,n_deciles+1)
    ranges, avg_tprs = [], []
    for i in range(n_deciles):
        low, high = bins[i], bins[i+1]
        mask = (mean_fpr>=low)&(mean_fpr<high)
        ranges.append(f'{low:.1f}-{high:.1f}')
        avg_tprs.append(_np.mean(mean_tpr[mask]))
        plt.axvline(i/n_deciles, ls='--', lw=0.5, alpha=0.4)

    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right'); plt.tight_layout(); plt.show()

    return pd.DataFrame({'FPR Range': ranges, 'Avg TPR': [round(x,3) for x in avg_tprs]})


def run_repeated_cv(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    n_repeats: int = 100,
    C: float = 20,
    kernel: str = 'rbf',
    random_state: int = 42
) -> Tuple[pd.DataFrame, float, float]:
    """
    Repeated stratified CV: returns per-sample misclassification stats and AUC summary.

    :returns: (misclassified_df, mean_auc, std_auc)
    """
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    grid = np.arange(0,1.01,0.01)

    aucs, tprs = [], []
    stats = pd.DataFrame(0, index=X.index, columns=['n_misclassified','sum_prob_wrong'])

    for train_idx, test_idx in cv.split(X, y):
        clf = SVC(C=C, kernel=kernel, probability=True, class_weight='balanced')
        clf.fit(X.iloc[train_idx], y.iloc[train_idx])
        probs = clf.predict_proba(X.iloc[test_idx])[:,1]
        aucs.append(roc_auc_score(y.iloc[test_idx], probs))

        fpr, tpr, _ = roc_curve(y.iloc[test_idx], probs)
        tprs.append(np.interp(grid, fpr, tpr))

        pred = (probs>=0.5).astype(int)
        wrong = pred != y.iloc[test_idx].values
        idxs = X.index[test_idx][wrong]
        stats.loc[idxs, 'n_misclassified'] += 1
        stats.loc[idxs, 'sum_prob_wrong'] += probs[wrong]

    iteration_means = np.array(aucs).reshape(n_repeats, n_splits).mean(axis=1)
    mean_auc = iteration_means.mean()
    std_auc = iteration_means.std()

    mask = stats['n_misclassified']>0
    stats.loc[mask,'avg_prob_wrong'] = stats.loc[mask,'sum_prob_wrong']/stats.loc[mask,'n_misclassified']
    misclassified_df = stats[mask].sort_values('n_misclassified', ascending=False)

    return misclassified_df, mean_auc, std_auc