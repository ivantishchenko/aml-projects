import numpy as np

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

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
X, y = util.load_train_data()
X_test = util.load_test_data()

# Do data processing
# 1. Replace NaNs by the colum means
__complete_matrix_colmean(X)
__complete_matrix_colmean(X_test)

# print(np.linalg.matrix_rank(X))
# print(np.linalg.matrix_rank(X_test))
#
# X = low_rank_approx(X, 4)
# X_test = low_rank_approx(X_test, 4)

# Validation and training split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.10, random_state=42)

# Do regression on the Validation data
reg = linear_model.LinearRegression()
reg.fit(X_train, y_train)
prediction_val = reg.predict(X_val)

# Calculate the metric
score = r2_score(y_val, prediction_val)
print("Coefficient of Determination = {}".format(score))

# Do regression on the Test data
prediction_test = reg.predict(X_test)

# Produce the CSV solution
util.produce_solution(prediction_test)
