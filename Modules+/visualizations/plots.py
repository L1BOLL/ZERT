# Modules 1.1/visualization/plots.py

from typing import List, Optional, Union, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore, ttest_ind
from statsmodels.stats.multitest import multipletests
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
import joblib
import json


def plot_heatmap(
    df: pd.DataFrame,
    markers: List[str],
    group_col: str = 'type',
    zscore_axis: int = 0,
    cmap: str = 'vlag',
    cluster_rows: bool = False,
    cluster_cols: bool = False,
    palette: Dict[Any, str] = {1: 'red', 0: 'blue'},
    figsize: tuple = (12, 8)
) -> None:
    """
    Plot a static heatmap of selected markers, z-score scaled per column.
    Rows colored by group.
    """
    # prepare data
    df_sorted = df.sort_values(group_col)
    colors = df_sorted[group_col].map(palette)
    data = df_sorted[markers]
    scaled = data.apply(zscore, axis=zscore_axis)
    scaled.index = df_sorted.index

    sns.clustermap(
        scaled,
        cmap=cmap,
        row_cluster=cluster_rows,
        col_cluster=cluster_cols,
        row_colors=colors,
        xticklabels=True,
        yticklabels=True,
        figsize=figsize
    )
    plt.title("Heatmap of Top Markers (z-score)")
    plt.show()


def plot_boxplots(
    df: pd.DataFrame,
    markers: List[str],
    group_col: str = 'type',
    top_n: Optional[int] = None,
    jitter: float = 0.2,
    strip_alpha: float = 0.4
) -> None:
    """
    For each marker, plot a boxplot and overlaid stripplot by group.
    """
    genes = markers if top_n is None else markers[:top_n]
    for gene in genes:
        plt.figure(figsize=(5, 4))
        sns.boxplot(x=group_col, y=gene, data=df)
        sns.stripplot(x=group_col, y=gene, data=df,
                      color='black', alpha=strip_alpha, jitter=jitter)
        plt.title(f'Expression of {gene} by {group_col}')
        plt.xlabel(f"{group_col}")
        plt.ylabel("Expression")
        plt.tight_layout()
        plt.show()


def plot_pca(
    X: Union[pd.DataFrame, np.ndarray],
    y: Optional[Union[pd.Series, np.ndarray]] = None,
    n_components: int = 2,
    palette: Dict[Any, str] = {1: 'red', 0: 'blue'},
    figsize: tuple = (8, 6)
) -> pd.DataFrame:
    """
    Perform PCA and scatter plot first two components.
    """
    if hasattr(X, 'values'):
        X_vals = X.values
    else:
        X_vals = X
    scaler = None
    # assume data is pre-scaled or handle scaling outside
    pca = PCA(n_components=n_components)
    pcs = pca.fit_transform(X_vals)
    df_pca = pd.DataFrame(pcs, columns=[f'PC{i+1}' for i in range(n_components)])
    if y is not None:
        df_pca['group'] = y.values if hasattr(y, 'values') else y
        df_pca.index = X.index
        plt.figure(figsize=figsize)
        sns.scatterplot(
            data=df_pca,
            x='PC1', y='PC2',
            hue='group',
            palette=palette,
            s=60, alpha=0.8
        )
        var = pca.explained_variance_ratio_
        plt.title(f'PCA (PC1 {var[0]*100:.1f}%, PC2 {var[1]*100:.1f}%)')
        plt.tight_layout()
        plt.show()
    return df_pca


def plot_tsne(
    X: Union[pd.DataFrame, np.ndarray],
    y: Optional[Union[pd.Series, np.ndarray]] = None,
    perplexity: int = 30,
    random_state: int = 42,
    init: str = 'pca',
    figsize: tuple = (8, 6),
    palette: Dict[Any, str] = {1: 'red', 0: 'blue'}
) -> pd.DataFrame:
    """
    Perform t-SNE and scatter plot.
    """
    X_vals = X.values if hasattr(X, 'values') else X
    tsne = TSNE(n_components=2, perplexity=perplexity,
                learning_rate='auto', init=init,
                random_state=random_state)
    res = tsne.fit_transform(X_vals)
    df_tsne = pd.DataFrame(res, columns=['Dim1', 'Dim2'])
    if y is not None:
        df_tsne['group'] = y.values if hasattr(y, 'values') else y
        df_tsne.index = X.index
        plt.figure(figsize=figsize)
        sns.scatterplot(
            data=df_tsne,
            x='Dim1', y='Dim2',
            hue='group', palette=palette,
            s=60, alpha=0.8
        )
        plt.title('t-SNE of Features')
        plt.tight_layout()
        plt.show()
    return df_tsne


def plot_umap(
    X: Union[pd.DataFrame, np.ndarray],
    y: Optional[Union[pd.Series, np.ndarray]] = None,
    n_components: int = 2,
    random_state: int = 42,
    figsize: tuple = (8, 6),
    palette: Dict[Any, str] = {1: 'red', 0: 'blue'}
) -> pd.DataFrame:
    """
    Perform UMAP and scatter plot.
    """
    X_vals = X.values if hasattr(X, 'values') else X
    um = umap.UMAP(n_components=n_components, random_state=random_state)
    res = um.fit_transform(X_vals)
    df_umap = pd.DataFrame(res, columns=[f'Dim{i+1}' for i in range(n_components)])
    if y is not None:
        df_umap['group'] = y.values if hasattr(y, 'values') else y
        df_umap.index = X.index
        plt.figure(figsize=figsize)
        sns.scatterplot(
            data=df_umap,
            x='Dim1', y='Dim2',
            hue='group', palette=palette,
            s=60, alpha=0.8
        )
        plt.title('UMAP of Features')
        plt.tight_layout()
        plt.show()
    return df_umap


def compute_volcano_data(
    df: pd.DataFrame,
    label_col: str = 'type'
) -> pd.DataFrame:
    """
    Compute log2 fold change and FDR-adjusted p-values for each feature.
    """
    results = []
    group1 = df[df[label_col] == 1]
    group0 = df[df[label_col] == 0]
    for gene in df.columns.drop(label_col):
        vals1 = group1[gene]
        vals0 = group0[gene]
        fc = vals1.mean() / vals0.mean() if vals0.mean() != 0 else np.nan
        log2fc = np.log2(fc) if fc and fc > 0 else 0.0
        _, pval = ttest_ind(vals1, vals0, equal_var=False)
        results.append({'gene': gene, 'log2FC': log2fc, 'p_value': pval})
    res_df = pd.DataFrame(results)
    _, pvals_fdr, _, _ = multipletests(res_df['p_value'], method='fdr_bh')
    res_df['p_fdr'] = pvals_fdr
    return res_df


def plot_volcano(
    df_vol: pd.DataFrame,
    log2fc_col: str = 'log2FC',
    pval_col: str = 'p_fdr',
    fc_thresh: float = 1.0,
    p_thresh: float = 0.05,
    figsize: tuple = (10, 8)
) -> None:
    """
    Plot a volcano chart with points colored by significance.
    """
    df = df_vol.copy()
    df['-log10(pval)'] = -np.log10(df[pval_col])
    df['color'] = 'gray'
    df.loc[(df[log2fc_col] >= fc_thresh) & (df[pval_col] < p_thresh), 'color'] = 'red'
    df.loc[(df[log2fc_col] <= -fc_thresh) & (df[pval_col] < p_thresh), 'color'] = 'blue'

    plt.figure(figsize=figsize)
    plt.scatter(df[log2fc_col], df['-log10(pval)'], c=df['color'], alpha=0.7, edgecolor='k')
    plt.axvline(fc_thresh, linestyle='--')
    plt.axvline(-fc_thresh, linestyle='--')
    plt.axhline(-np.log10(p_thresh), linestyle='--')
    plt.xlabel('Log2 Fold Change')
    plt.ylabel('-Log10(FDR-adjusted p-value)')
    plt.title('Volcano Plot')
    plt.tight_layout()
    plt.show()


def evaluate_model(
    model: Any,
    X_test: Union[pd.DataFrame, np.ndarray],
    y_test: Union[pd.Series, np.ndarray]
) -> Dict[str, Any]:
    """
    Compute test set AUC, accuracy, and plot ROC.
    """
    if hasattr(X_test, 'values'):
        X_vals = X_test.values
    else:
        X_vals = X_test
    if hasattr(y_test, 'values'):
        y_vals = y_test.values
    else:
        y_vals = y_test

    probs = model.predict_proba(X_vals)[:, 1]
    auc = roc_auc_score(y_vals, probs)
    acc = accuracy_score(y_vals, model.predict(X_vals))

    fpr, tpr, _ = roc_curve(y_vals, probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Test ROC Curve')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()

    return {'test_auc': auc, 'test_accuracy': acc}


def save_model(
    model: Any,
    path: str
) -> None:
    """
    Persist a model to disk via joblib.
    """
    joblib.dump(model, path)


def save_features(
    features: List[str],
    path: str
) -> None:
    """
    Save selected feature list to JSON.
    """
    with open(path, 'w') as f:
        json.dump(features, f)