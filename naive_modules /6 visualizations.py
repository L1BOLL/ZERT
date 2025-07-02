from scipy.stats import zscore

heat_df = filtered_df.sort_values("type")

type_colors = heat_df["type"].map({1: 'red', 0: 'blue'})  # type = red, control = blue

top_genes = bmg[:]  # or bmg[:N] if you want top N
heat_data = heat_df[top_genes]

scaled_data = heat_data.apply(zscore, axis=0)

scaled_data.index = heat_df.index  # make sure sample alignment is correct

sns.clustermap(
    scaled_data,
    cmap="vlag",
    col_cluster=False,
    row_cluster=False,  # <- stop it from reordering rows!
    row_colors=type_colors,
    xticklabels=True,
    yticklabels=True,
    figsize=(12, 8)
)

plt.title("Heatmap of Top Biomarker Transcripts (z-score scaled)")
plt.show()


# Box plots of each feature
top_20 = bmg[:20]

for gene in top_20:
    plt.figure(figsize=(5, 4))
    sns.boxplot(x='type', y=gene, data=filtered_df)
    sns.stripplot(x='type', y=gene, data=filtered_df, color='black', alpha=0.4, jitter=0.2)
    plt.title(f'Expression of {gene} by Class')
    plt.xlabel("Class (0 = Control, 1 = type)")
    plt.ylabel("Expression")
    plt.tight_layout()
    plt.show()


# PCA
from sklearn.decomposition import PCA



scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)


pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
pca_df['type'] = y.values  # Add class labels
pca_df.index = X.index     # (optional) retain sample names


plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=pca_df,
    x='PC1', y='PC2',
    hue='type',
    palette={1: 'red', 0: 'blue'},
    s=60,
    alpha=0.8
)

plt.title('PCA of Biomarker Expression')
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
plt.legend(title='Class')
plt.tight_layout()
plt.show()


# t-SNE

from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, perplexity=30, learning_rate='auto', init='pca', random_state=42)
tsne_result = tsne.fit_transform(X_scaled)

tsne_df = pd.DataFrame(tsne_result, columns=['Dim1', 'Dim2'])
tsne_df['type'] = y.values
tsne_df.index = X.index

plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=tsne_df,
    x='Dim1', y='Dim2',
    hue='type',
    palette={1: 'red', 0: 'blue'},
    s=60, alpha=0.8
)
plt.title('t-SNE of Biomarker Expression')
plt.legend(title='Class')
plt.tight_layout()
plt.show()


# UMAP
import umap

umap_model = umap.UMAP(n_components=2, random_state=42)
umap_result = umap_model.fit_transform(X_scaled)

umap_df = pd.DataFrame(umap_result, columns=['Dim1', 'Dim2'])
umap_df['type'] = y.values
umap_df.index = X.index

plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=umap_df,
    x='Dim1', y='Dim2',
    hue='type',
    palette={1: 'red', 0: 'blue'},
    s=60, alpha=0.8
)
plt.title('UMAP of Biomarker Expression')
plt.legend(title='Class')
plt.tight_layout()
plt.show()


from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
# === Compute stats per gene ===
def compute_volcano_data(df, label_col="type"):
    results = []

    types = df[df[label_col] == 1]
    controls = df[df[label_col] == 0]

    for gene in df.columns.drop(label_col):
        type_vals = types[gene]
        ctrl_vals = controls[gene]

        # Compute log2 fold change
        fc = type_vals.mean() / ctrl_vals.mean()
        log2fc = np.log2(fc) if fc > 0 else 0

        # t-test
        _, p = ttest_ind(type_vals, ctrl_vals, equal_var=False)
        results.append({"gene": gene, "log2FC": log2fc, "p_value": p})

    df_results = pd.DataFrame(results)

    # Add FDR-adjusted p-values
    _, pvals_fdr, _, _ = multipletests(df_results["p_value"], method="fdr_bh")
    df_results["p_fdr"] = pvals_fdr

    return df_results

# === Volcano plot ===
def plot_volcano(df, log2fc_col="log2FC", pval_col="p_fdr", fc_thresh=1, p_thresh=0.05):
    df = df.copy()
    df["-log10(pval)"] = -np.log10(df[pval_col])

    df["color"] = "gray"
    df.loc[(df[log2fc_col] >= fc_thresh) & (df[pval_col] < p_thresh), "color"] = "red"
    df.loc[(df[log2fc_col] <= -fc_thresh) & (df[pval_col] < p_thresh), "color"] = "blue"

    plt.figure(figsize=(10, 8))
    plt.scatter(df[log2fc_col], df["-log10(pval)"], c=df["color"], alpha=0.7, edgecolor='k')
    plt.axvline(x=fc_thresh, linestyle='--', color='black')
    plt.axvline(x=-fc_thresh, linestyle='--', color='black')
    plt.axhline(y=-np.log10(p_thresh), linestyle='--', color='black')

    plt.xlabel("Log2 Fold Change")
    plt.ylabel("-Log10(FDR-adjusted p-value)")
    plt.title("Volcano Plot (FDR Controlled)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


volcano_df = compute_volcano_data(filtered_df)
plot_volcano(volcano_df)

# Saving the SVM model
import joblib
#joblib.dump(model, '28may25.pkl')

# Saving the bmg

import json
#with open("top_features.json", "w") as f:
#    json.dump(bmg, f)

X_test = test_set[bmg]
y_test = test_set["type"]

# External Validation
from sklearn.metrics import accuracy_score


y_pred_prob = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

test_auc = roc_auc_score(y_test, y_pred_prob)
test_accuracy = accuracy_score(y_test, y_pred)

print("Test AUC:", roc_auc_score(y_test, y_pred_prob))
print("Test Accuracy:", accuracy_score(y_test, y_pred))

fpr, tpr, _ = roc_curve(y_test, y_pred_prob)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {test_auc:.3f}")
plt.plot([0, 1], [0, 1], 'k--', label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve on Test Set")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()
