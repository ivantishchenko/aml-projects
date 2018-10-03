import numpy as np

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn import preprocessing
from sklearn import neighbors
from sklearn import feature_selection
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
# Constants
import util_data as util


def __complete_matrix_colmean(X):
    """
    Complete missing values in the data matrix
    """

    # get col means
    col_mean = np.nanmean(X, axis=0)
    # Find indicies that you need to replace
    idxs = np.where(np.isnan(X))
    X[idxs] = np.take(col_mean, idxs[1])


def __complete_matrix_zeros(X):
    """
    Completion with zeros
    """

    idxs = np.isnan(X)
    X[idxs] = 0


def low_rank_approx(A=None, r=1):
    """
    Computes an r-rank approximation of a matrix
    given the component u, s, and v of it's SVD
    Requires: numpy
    """
    SVD = np.linalg.svd(A)
    u, s, v = SVD
    Ar = np.zeros((len(u), len(v)))
    for i in range(r):
        Ar += s[i] * np.outer(u.T[i], v[i])
    return Ar


'''
Solution 1
'''

# Load data
print("1. Loading the data")
X, y = util.load_train_data()
X_test = util.load_test_data()

# Do data processing
# 1. Replace NaNs by the colum means
print("2. Starting preprocessing the data")
__complete_matrix_colmean(X)
__complete_matrix_colmean(X_test)

# 2. Zero Mean, Unit Variance
scaler = preprocessing.StandardScaler().fit(X)
X = scaler.transform(X)
X_test = scaler.transform(X_test)

# 3. Outlier detection
# lof = neighbors.LocalOutlierFactor(n_neighbors=20, contamination=0.1)
# outliers = lof.fit_predict(X)
#
# unique, counts = np.unique(outliers, return_counts=True)
# count_dict = dict(zip(unique, counts))
#
# X = X[outliers == 1]
# y = y[outliers == 1]

# 4. Feature selection
# sel = feature_selection.VarianceThreshold(threshold=(.8 * (1 - .8)))
# X = sel.fit_transform(X)
# X_test = sel.fit_transform(X_test)

# Validation and training split K Folds
print("3. Doing validation and training split")
# kf = KFold(n_splits=10)
# for train_index, val_index in kf.split(X):
#     print("TRAIN:", train_index, "TEST:", val_index)
#     X_train, X_val = X[train_index], X[val_index]
#     y_train, y_val = y[train_index], y[val_index]

print("4. Performing regression")
reg = linear_model.ElasticNet()
reg_scores = cross_val_score(reg, X, y, cv=10, scoring='r2')

# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Do regression on the Validation data

# Calculate the metric
print("5. Calculating the score")
# score = r2_score(y_val, prediction_val)
print("N Scores are = {}".format(reg_scores))
score = np.mean(reg_scores)
print("Averaged coefficient of Determination = {}".format(score))
std_scores = np.std(reg_scores)
print("STD of N scores = {}".format(std_scores))

print("6. Training on the whole set")
reg.fit(X, y)

print("7. Generating predictions")
prediction_test = reg.predict(X_test)
# Produce the CSV solution
util.produce_solution(prediction_test)
