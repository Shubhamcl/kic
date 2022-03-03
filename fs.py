import sys
from time import time

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import RidgeCV
from sklearn.datasets import load_diabetes
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SequentialFeatureSelector

# [[ Get Dataset ]]
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target

# [[ Feature selection with Ridge Regression ]]

# Get feature importance from Ridge Regression
ridge = RidgeCV(alphas=np.logspace(-6, 6, num=5)).fit(X, y)
importance = np.abs(ridge.coef_)
feature_names = np.array(diabetes.feature_names)

# Feature selection
threshold = np.sort(importance)[-3] + 0.01
# tic = time()
sfm = SelectFromModel(ridge, threshold=threshold).fit(X, y)
# toc = time()
print(f"Features selected by SelectFromModel: {feature_names[sfm.get_support()]}")
# print(f"Done in {toc - tic:.3f}s")


# [[ Sequential feature selection (forward and backward) ]]
# Also uses Ridge regression

tic_fwd = time()
sfs_forward = SequentialFeatureSelector(
    ridge, n_features_to_select=2, direction="forward"
).fit(X, y)
toc_fwd = time()

tic_bwd = time()
sfs_backward = SequentialFeatureSelector(
    ridge, n_features_to_select=2, direction="backward"
).fit(X, y)
toc_bwd = time()

print(
    "Features selected by forward sequential selection: "
    f"{feature_names[sfs_forward.get_support()]}"
)
print(f"Done in {toc_fwd - tic_fwd:.3f}s")

print(
    "Features selected by backward sequential selection: "
    f"{feature_names[sfs_backward.get_support()]}"
)
print(f"Done in {toc_bwd - tic_bwd:.3f}s")


# Notes:
# how to apply forward selection on this?
# Same way, need to apply pearson correlation on it!?