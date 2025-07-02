# ### Model Training without the 5% misclassified samples

# 0.  get the indices of the 52 “bad” rows
bad_idx = misclassified_df.loc[misclassified_df["n_misclassified"] >= 50].index            # every row in misclassified_df is one you flagged

# 1.  drop them from the full data
X_clean = X.drop(index=bad_idx)
y_clean = y.drop(index=bad_idx)

# 2.  rerun the exact same CV loop on the cleaned data
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=100, random_state=42)
clf_template = SVC(C=20, kernel="rbf", probability=True)

auc_folds, tprs, stats = [], [], pd.DataFrame(
    0, index=X_clean.index,
    columns=["n_misclassified", "sum_prob_wrong"]
).astype({"n_misclassified": int, "sum_prob_wrong": float})

for tr, te in cv.split(X_clean, y_clean):
    clf = clf_template.fit(X_clean.iloc[tr], y_clean.iloc[tr])

    prob_pos = clf.predict_proba(X_clean.iloc[te])[:, 1]
    y_test   = y_clean.iloc[te].values
    pred     = (prob_pos >= 0.5).astype(int)

    auc_folds.append(roc_auc_score(y_test, prob_pos))

    fpr, tpr, _ = roc_curve(y_test, prob_pos)
    tprs.append(np.interp(grid, fpr, tpr))

    wrong = pred != y_test
    w_idx = X_clean.index[te][wrong]
    stats.loc[w_idx, "n_misclassified"] += 1
    stats.loc[w_idx, "sum_prob_wrong"] += prob_pos[wrong]

# final model on the cleaned, full dataset
final_clf = SVC(C=20, kernel="rbf", probability=True).fit(X_clean, y_clean)

# save or use it:
joblib.dump(final_clf, "svm_after_pruning.joblib")  

print("""Pooled ROC:

How built: Dump all test-fold predictions into one long vector, run roc_curve once.

Meaning: Treats every observation as if it came from a single test set.

Pros: Full resolution; single clean step curve; AUC (area-under-curve) equals micro-averaged performance.

Cons: Hides fold-to-fold variance; optimistic if folds overlap subjects (data leakage risk).\n  \n""")
plot_mean_roc_with_percentiles(mean_fpr, mean_tpr, mean_auc)

print(""" Average (fold-wise mean) ROC:

How built: Compute a ROC for each CV (cross-validation) fold, interpolate on a common FPR (false-positive-rate) grid, then average TPR (true-positive-rate) point-wise; report mean ± SD (standard-deviation).

Meaning: “Expected” ROC of an arbitrary model clone trained on similar data.

Pros: Shows variability; lets you draw ±SD or CI (confidence-interval) bands.

Cons: Needs interpolation; resolution limited by chosen grid; mean AUC differs slightly from pooled’s micro-AUC. \n\n""")
decile_df = plot_mean_roc_with_deciles(grid, tprs, mean_auc)

# ### RFE

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold

cv      = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
k_list  = range(1, 21)
acc_cv  = []
acc_std = []

for k in k_list:
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("sel",   SelectKBest(mutual_info_classif, k=k)),
        ("svc",   SVC(C=20, kernel="rbf", probability=True, class_weight='balanced'))
    ])
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    acc_cv.append(scores.mean())
    acc_std.append(scores.std())

acc_cv  = np.array(acc_cv)
acc_std = np.array(acc_std)

best_k  = k_list[acc_cv.argmax()]
best_acc = acc_cv.max()

plt.figure(figsize=(8,5))
plt.plot(k_list, acc_cv, marker='o')
plt.fill_between(k_list, acc_cv-acc_std, acc_cv+acc_std, alpha=0.2)
plt.axvline(best_k, ls='--', color='red')

# force integer ticks
ax = plt.gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.xlabel("# of biomarkers")
plt.ylabel("Cross-validated AUC")
plt.title("Learning curve")
plt.grid(True)
plt.tight_layout()
plt.show()