import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# # Feature selection

filtered_df = df.set_index('Transcript')
X = filtered_df.drop(columns=["type"])
y = filtered_df["type"]

# Normalize
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler(with_mean=False)
X_scaled = scaler.fit_transform(X)



# LASSO Feature Selection

from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold

lasso = LogisticRegressionCV(
    penalty='l1',
    solver='saga',
    cv=StratifiedKFold(5),
    scoring='roc_auc',
    max_iter=5000
)
lasso.fit(X_scaled, y)

coef = lasso.coef_[0]

lasso_selected = [gene for gene, c in zip(X.columns, coef) if c != 0]
X_lasso = X[lasso_selected]

# Boruta to find all relevant genes

from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
boruta = BorutaPy(rf, n_estimators='auto', max_iter=100)
boruta.fit(X.values, y.values)

boruta_selected = X.columns[boruta.support_]
X_boruta = X[boruta_selected]

# Stability Selection

""" from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression

counts = np.zeros(X.shape[1])
for _ in range(100):
    X_res, y_res = resample(X, y)
    clf = LogisticRegression(penalty='l1', solver='saga', max_iter=5000)
    clf.fit(X_res, y_res)
    counts += (clf.coef_[0] != 0)

stability_selected = X.columns[counts > 80]
X_stable = X[stability_selected] """

# Model Evaluation

final_features = set(lasso_selected) & set(boruta_selected)
X_final = X[list(final_features)]

if len(final_features) < 20:
    from collections import Counter

    # Combine all selected feature lists
    all_selected = list(lasso_selected) + list(boruta_selected)
    feature_counts = Counter(all_selected)
    
    # Keep features selected by at least two methods
    final_features = [f for f, count in feature_counts.items() if count >= 2]
    X_final = X[final_features]

X_final["type"] = df["type"].tolist()
filtered_df = X_final