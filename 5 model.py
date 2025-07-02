# X and y for training : biomarker group dataframe (X) & type column from df (y)
X = filtered_df[bmg]
y = filtered_df.type

X

y

# #### Run from here to change train_test_split for other combinations

# Train Test Split Internal validation
from sklearn.model_selection import train_test_split, StratifiedKFold
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify = df["type"])

# training and getting score of SVM model

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# SVM Kernels : 'sigmoid', 'rbf', 'linear', 'poly' // The best one so far is: 'rbf'
model = make_pipeline(StandardScaler(), SVC(C=20, kernel = "rbf", probability=True, class_weight='balanced'))
model.fit(X_train, y_train)

model.score(X_test, y_test)

# Getting SVM hyperparameters
params = model.get_params()
params

y_score = model.predict_proba(X_test)[:, 1]
y_score


def run_cross_validated_model_with_counts(X, y, C=20, kernel="rbf", plot=True, ci=0.99):
    """
    Everything in the original run_cross_validated_model, plus:
    -----------------------------------------------------------------
    * `misclassification_counts` – a DataFrame showing, for every row
      of X, how many of the 5 folds it was mis-predicted (0-5).
    * The original return structure and printed output are untouched.
    """
    # --------------------------- originals ---------------------------
    from sklearn.model_selection import StratifiedKFold
    from sklearn.svm import SVC
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_curve, auc, confusion_matrix
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy.stats import norm

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    folds_auc, folds_confusion, misclassified_samples = [], [], []
    mean_fpr = np.linspace(0, 1, 100)
    tprs_interp = []

    # NEW ── counter for “how many times was each sample wrong?”
    mis_count = pd.Series(0, index=X.index, dtype=int, name="Times Misclassified")

    if plot:
        plt.figure(figsize=(10, 8))

    for i, (train_idx, test_idx) in enumerate(kf.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        svm = make_pipeline(StandardScaler(),
                            SVC(C=C, kernel=kernel,
                                probability=True, class_weight='balanced'))
        svm.fit(X_train, y_train)

        y_score = svm.predict_proba(X_test)[:, 1]
        y_pred  = svm.predict(X_test)

        # ---------- original ROC/AUC bookkeeping ----------
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        folds_auc.append(roc_auc)

        tpr_interp = np.interp(mean_fpr, fpr, tpr)
        tpr_interp[0] = 0.0
        tprs_interp.append(tpr_interp)

        if plot:
            plt.plot(fpr, tpr, lw=1.5, alpha=0.6,
                     label=f"Fold {i+1} (AUC = {roc_auc:.2f})")

        cm = confusion_matrix(y_test, y_pred)
        folds_confusion.append(cm)

        # ---------- original mis-classified rows ----------
        mis_mask = y_pred != y_test
        mis_df = pd.DataFrame({
            "Index": X_test.index[mis_mask],
            "True Label": y_test[mis_mask].values,
            "Predicted Label": y_pred[mis_mask],
            "Predicted Prob": y_score[mis_mask]
        })
        misclassified_samples.append(mis_df)

        # ---------- NEW count increment ----------
        mis_count.loc[X_test.index[mis_mask]] += 1

    # --------------------------- originals (CI, plots, printing) ---------------------------
    tprs_interp = np.array(tprs_interp)
    mean_tpr = np.mean(tprs_interp, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    lower_bound = np.percentile(tprs_interp, ((1 - ci) / 2) * 100, axis=0)
    upper_bound = np.percentile(tprs_interp, (1 - (1 - ci) / 2) * 100, axis=0)

    if plot:
        plt.fill_between(mean_fpr, lower_bound, upper_bound,
                         color='grey', alpha=0.3, label=f"{int(ci*100)}% CI")
        plt.plot([0, 1], [0, 1], 'k--', lw=1.5)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("5-Fold ROC Curves with Mean AUC and Confidence Interval")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.show()

    print("\nFold AUCs")
    for i, auc_score in enumerate(folds_auc):
        print(f"Fold {i+1}: AUC = {auc_score:.4f}")
    print(f"Mean AUC: {mean_auc:.4f}")

    print("\nConfusion Matrices")
    for i, cm in enumerate(folds_confusion):
        print(f"Fold {i+1}:\n{cm}\n")

    cm_avg = np.mean(folds_confusion, axis=0).round(2)
    print("Mean Confusion Matrix:\n", cm_avg)

    # ---------- original mis-classified list ----------
    all_misclassified = (
        pd.concat(misclassified_samples)
          .reset_index(drop=True)
          .sort_values(by="Predicted Prob", ascending=False)
          .reset_index(drop=True)
    )
    print("\nMisclassified Samples Summary:")
    print(all_misclassified)

    # ---------- NEW: summary of counts over the 5 folds ----------
    mis_count_df = (
        mis_count.rename("Times Misclassified")
                 .reset_index()
                 .rename(columns={"index": "Index"})
                 .sort_values("Times Misclassified", ascending=False)
                 .reset_index(drop=True)
    )
    print("\nMisclassification Counts across 5 folds:")
    print(mis_count_df[mis_count_df["Times Misclassified"] > 0])

    # ---------- original AUC CI ----------
    auc_std = np.std(folds_auc)
    z = norm.ppf(0.95)
    auc_ci_lower = np.mean(folds_auc) - z * auc_std / np.sqrt(len(folds_auc))
    auc_ci_upper = np.mean(folds_auc) + z * auc_std / np.sqrt(len(folds_auc))

    # ---------- original probs bookkeeping ----------
    # (re-using all_misclassified’s concat logic keeps code concise)
    all_probs = (
        pd.concat([df[["Index", "True Label", "Predicted Prob"]]
                   for df in misclassified_samples], ignore_index=True)
        .sort_values("Index")
        .reset_index(drop=True)
    )

    # --------------------------- return ---------------------------
    return (
        {
            "mean_auc":           mean_auc,
            "fold_aucs":          folds_auc,
            "auc_ci":             (auc_ci_lower, auc_ci_upper),
            "confusion_matrices": folds_confusion,
            "mean_confusion":     cm_avg,
            "mean_tpr":           mean_tpr,
            "mean_fpr":           mean_fpr,
            "tprs_interp":        tprs_interp,
            "ci_lower":           lower_bound,
            "ci_upper":           upper_bound,
            "misclassified_samples":   all_misclassified,
            "misclassification_counts": mis_count_df,   # <── NEW
            "all_predicted_probs":     all_probs,
        },
        folds_confusion  # unchanged second return
    )



# run with misclassified ordered

results, folds_confusion = run_cross_validated_model_with_counts(X, y)

results


def plot_mean_roc_with_deciles(mean_fpr, mean_tpr, mean_auc, n_deciles=10):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    plt.figure(figsize=(10, 8))
    plt.plot(mean_fpr, mean_tpr, color='black', lw=2.5, label=f"Mean ROC (AUC = {mean_auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5)

    for i in range(1, n_deciles):
        plt.axvline(x=i / n_deciles, linestyle='--', color='grey', lw=0.5, alpha=0.4)

    decile_bins = np.linspace(0, 1, n_deciles + 1)
    decile_tprs = []
    decile_ranges = []

    for i in range(n_deciles):
        start, end = decile_bins[i], decile_bins[i + 1]
        mask = (mean_fpr >= start) & (mean_fpr < end)
        tpr_vals = mean_tpr[mask]
        avg_tpr = np.mean(tpr_vals) if len(tpr_vals) > 0 else np.nan
        decile_tprs.append(avg_tpr)
        decile_ranges.append(f"{start:.1f}-{end:.1f}")

    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("Mean ROC Curve with Decile Trade-offs")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    decile_df = pd.DataFrame({
        "FPR Range": decile_ranges,
        "Avg TPR": [round(x, 3) if not np.isnan(x) else "-" for x in decile_tprs]
    })

    print("\nROC Decile Trade-offs Summary:")
    print(decile_df.to_string(index=False))

    return decile_df



# 5FCV run deciles
mean_fpr = results["mean_fpr"]
mean_tpr = results["mean_tpr"]
mean_auc = results["mean_auc"]

plot_mean_roc_with_deciles(mean_fpr, mean_tpr, mean_auc)


def plot_mean_roc_with_percentiles(mean_fpr, mean_tpr, mean_auc, n_percentiles=100, excel_path='roc_percentile_summary.xlsx'):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    plt.figure(figsize=(12, 8))
    plt.plot(mean_fpr, mean_tpr, color='black', lw=2.5, label=f"Mean ROC (AUC = {mean_auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5)

    for i in range(1, n_percentiles):
        plt.axvline(x=i / n_percentiles, linestyle='--', color='grey', lw=0.3, alpha=0.3)

    percentile_bins = np.linspace(0, 1, n_percentiles + 1)
    percentile_tprs = []
    percentile_ranges = []

    for i in range(n_percentiles):
        start, end = percentile_bins[i], percentile_bins[i + 1]
        mask = (mean_fpr >= start) & (mean_fpr < end)
        tpr_vals = mean_tpr[mask]
        avg_tpr = np.mean(tpr_vals) if len(tpr_vals) > 0 else np.nan
        percentile_tprs.append(avg_tpr)
        percentile_ranges.append(f"{start:.2f}-{end:.2f}")

    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("Mean ROC Curve with Percentile Trade-offs")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    percentile_df = pd.DataFrame({
        "FPR Range": percentile_ranges,
        "Avg TPR": [round(x, 3) if not np.isnan(x) else "-" for x in percentile_tprs]
    })

    # Save to Excel
    #percentile_df.to_excel(excel_path, index=False)

    print(f"\nROC Percentile Trade-offs Summary saved to: {excel_path}")
    return percentile_df


# 5FCV run percentiles
plot_mean_roc_with_percentiles(mean_fpr, mean_tpr, mean_auc)

# Sum confusion matrix

cm_sum = np.sum(folds_confusion, axis=0)

plt.figure(figsize=(5, 4))
sns.heatmap(cm_sum, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.title('Average Confusion Matrix (5-Fold CV)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

import shap
explainer = shap.Explainer(model.predict_proba, X_train, feature_names=X_train.columns)

shap_values = explainer(X_test)

shap_matrix = shap_values.values[:, :, 1]

shap_df = pd.DataFrame(
    shap_matrix,
    columns=X_test.columns,
    index=X_test.index
)

print(shap_matrix.shape)      # should be (n_samples, n_features)
print(len(X_test.columns))    # should match shap_matrix.shape[1]

mean_shap = shap_df.abs().mean().sort_values(ascending=False)
top_genes = mean_shap.head(10)

print("Top transcripts by SHAP importance:")
print(top_genes)

plt.figure(figsize=(10, 6))
top_genes[::-1].plot(kind='barh', color='salmon')
plt.xlabel("Mean |SHAP value|")
plt.title("Top Transcripts Driving SVM Predictions")
plt.grid(True)
plt.tight_layout()
plt.show()


# ### For testing the 5FCV 100 times

"""
1. 100×5-fold **CV** (cross-validation) with **SVC** (support-vector-classifier)
2. Mean & std **AUC** (area-under-curve) across the 100 repeats
3. `misclassified_df` for every row: how many times mis-predicted + mean wrong-class probability
4. **ROC** (receiver-operating-characteristic) plot of mean curve plus variability:
   * shaded ±1 SD
   * optional boxplots of the full **TPR** (true-positive-rate) distribution at
     each fixed **FPR** (false-positive-rate) grid-point
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.svm import SVC                    # support-vector-classifier
from sklearn.metrics import roc_auc_score, roc_curve

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=100, random_state=42)
model_kwargs = dict(C=20, kernel="rbf", probability=True)

# storage for results
auc_folds = []                      # AUC of every one of the 500 test splits
grid = np.arange(0.0, 1.01, 0.01)    # fixed FPR points (0–1 step 0.1)
tprs = []                           # interpolated TPRs for each split (500 × 11)

stats = pd.DataFrame(               # per-sample misclassification bookkeeping
    0, index=X.index,
    columns=["n_misclassified", "sum_prob_wrong"]
).astype({"n_misclassified": int, "sum_prob_wrong": float})


for train_idx, test_idx in cv.split(X, y):
    clf = SVC(**model_kwargs).fit(X.iloc[train_idx], y.iloc[train_idx])

    prob_pos = clf.predict_proba(X.iloc[test_idx])[:, 1]
    y_test   = y.iloc[test_idx].values
    pred     = (prob_pos >= 0.5).astype(int)

    # 1 ) AUC
    auc_folds.append(roc_auc_score(y_test, prob_pos))

    # 2 ) ROC sampling
    fpr, tpr, _ = roc_curve(y_test, prob_pos)
    tprs.append(np.interp(grid, fpr, tpr))      # interpolate TPR at fixed FPR grid

    # 3 ) misclassification stats
    wrong_mask = pred != y_test
    mis_idx = X.index[test_idx][wrong_mask]      # original indices of wrongly-predicted rows
    stats.loc[mis_idx, "n_misclassified"] += 1
    stats.loc[mis_idx, "sum_prob_wrong"] += prob_pos[wrong_mask]


# 3a. AUC summary (reshape 500×5 ⇒ 100×5 to get repeat-level means)
auc_folds = np.array(auc_folds)
iteration_means = auc_folds.reshape(100, 5).mean(axis=1)

print(f"Mean AUC over 100×5-fold CV : {iteration_means.mean():.4f}")
print(f"Std dev of those 100 means  : {iteration_means.std():.4f}")

# 3b. Per-sample misclassification DataFrame
mask = stats["n_misclassified"] > 0
stats.loc[mask, "avg_prob_wrong"] = (
    stats.loc[mask, "sum_prob_wrong"] / stats.loc[mask, "n_misclassified"]
)
misclassified_df = stats.loc[mask, ["n_misclassified", "avg_prob_wrong"]] \
                        .sort_values("n_misclassified", ascending=False)

print("\nmisclassified_df head():")
print(misclassified_df.head())

def plot_mean_roc_with_deciles(mean_fpr, tprs_all, mean_auc=None, n_deciles=100):
    """
    Draws the mean ROC curve, a ±1-SD band, and a boxplot of the TPR
    distribution at each FPR decile. Also prints a decile table with
    mean and std TPR.

    Parameters
    ----------
    mean_fpr   : 1-D array of the common FPR grid (e.g. np.arange(0,1.01,0.1))
    tprs_all   : 2-D array, shape (n_runs, len(mean_fpr)), interpolated TPRs
    mean_auc   : float, pre-computed AUC of the mean ROC (optional)
    n_deciles  : int, number of vertical slices (default 10 ⇒ 0.1 steps)

    Returns
    -------
    pandas.DataFrame with 'FPR Range', 'Avg TPR', 'Std TPR'
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import auc as _auc

    tprs_all = np.asarray(tprs_all)
    mean_tpr = tprs_all.mean(axis=0)
    std_tpr  = tprs_all.std(axis=0)
    upper    = np.clip(mean_tpr + std_tpr, 0, 1)
    lower    = np.clip(mean_tpr - std_tpr, 0, 1)

    if mean_auc is None:
        mean_auc = _auc(mean_fpr, mean_tpr)

    plt.figure(figsize=(10, 8))
    plt.plot(mean_fpr, mean_tpr, lw=2.5, color='black',
             label=f"Mean ROC (AUC = {mean_auc:.3f})")
    plt.plot(mean_fpr, upper, '--', lw=1.5, color='red',  label='Mean +1 SD')
    plt.plot(mean_fpr, lower, '--', lw=1.5, color='blue', label='Mean −1 SD')
    plt.fill_between(mean_fpr, lower, upper, color='grey', alpha=0.15)


    decile_bins = np.linspace(0, 1, n_deciles + 1)
    
    decile_ranges    = []   # labels for each slice
    avg_tpr          = []   # mean TPR in each slice
    std_tpr_decile   = []   # SD  TPR in each slice
    
    for i in range(n_deciles):
        start, end = decile_bins[i], decile_bins[i + 1]
        mask = (mean_fpr >= start) & (mean_fpr < end)
        data = tprs_all[:, mask].ravel()
    
        decile_ranges.append(f"{start:.1f}-{end:.1f}")
        avg_tpr.append(data.mean() if data.size else np.nan)
        std_tpr_decile.append(data.std() if data.size else np.nan)
        
    # optional vertical decile lines
    for i in range(1, n_deciles):
        plt.axvline(i / n_deciles, ls='--', lw=0.7, color='grey', alpha=0.4)

    plt.plot([0, 1], [0, 1], 'k--', lw=1.3)
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("Mean ROC with ±1 SD Curves")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()



    
    # summary table
    import pandas as pd
    decile_df = pd.DataFrame({
        "FPR Range": [f"{decile_bins[i]:.1f}-{decile_bins[i+1]:.1f}"
                      for i in range(n_deciles)],
        "Avg TPR":   [round(x, 3) if np.isfinite(x) else "-" for x in avg_tpr],
        "Std TPR":   [round(x, 3) if np.isfinite(x) else "-" for x in std_tpr_decile]
    })
    print("\nDecile Summary:")
    print(decile_df.head(10))
    return decile_df



decile_df = plot_mean_roc_with_deciles(grid, tprs, mean_auc)



print(misclassified_df.shape)
misclassified_df

decile_df