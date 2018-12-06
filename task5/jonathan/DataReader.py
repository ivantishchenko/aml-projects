import pandas as pd
import numpy as np

def read_data(filename):
    csv_data = pd.read_csv(filename)
    values = csv_data.values
    return values[:, 1:], values[:, 0]

def replace_nans(X, value):
    inds = np.where(np.isnan(X))
    X[inds] = value
    return X

def normalize_data(X, mean=None, std=None):
    if mean is None:
        mean = X.mean(axis=0)
    if std is None:
        std = X.std(axis=0)

    X = (X - mean) / std
    return X, mean, std