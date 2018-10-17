import numpy as np

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn import preprocessing
from sklearn import neighbors
from sklearn import feature_selection
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn import impute
from sklearn import ensemble
from sklearn import neural_network
from sklearn import svm
from sklearn import kernel_ridge
from sklearn import model_selection
from sklearn import tree

# Constants
import util_data as util

'''
Solution 1
'''

# Load data
print("1. Loading the data\n")
X, y = util.load_train_data()
X_test = util.load_test_data()
y = y.ravel()

pass