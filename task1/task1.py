import csv
import numpy as np

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Constants
TRAIN_M = 1212
TRAIN_N = 887

TEST_M = 776
TEST_N = 887

# Start of the helper functions

'''
Load the test data
'''


def __load_test_data():
    # Load X_test
    with open('X_test.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        feature_string_matrix = []
        for row in reader:
            feature_list = []
            for i in range(TEST_N):
                x_value = row['x' + str(i)]
                # Hit missing values
                if x_value == '':
                    feature_list.append(np.nan)
                else:
                    feature_list.append(float(row['x' + str(i)]))
            feature_string_matrix.append(feature_list)
        X_test = np.array(feature_string_matrix)
        return X_test


'''
Load the train data
'''


def __load_train_data():
    # Load X_train
    with open('X_train.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        feature_string_matrix = []
        for row in reader:
            feature_list = []
            for i in range(TRAIN_N):
                x_value = row['x' + str(i)]
                # Hit missing values
                if x_value == '':
                    feature_list.append(np.nan)
                else:
                    feature_list.append(float(row['x' + str(i)]))
            feature_string_matrix.append(feature_list)
        X_train = np.array(feature_string_matrix)
    # Load Y_train
    with open('y_train.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        y_string = []
        for row in reader:
            y_value = [float(row['y'])]
            y_string.append(y_value)
        y_train = np.array(y_string)
    return X_train, y_train


'''
Produce the CSV of a solution
'''


def __produce_solution(y):
    with open('out.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', lineterminator="\n")
        writer.writerow(['id', 'y'])
        for i in range(y.shape[0]):
            writer.writerow([float(i), y[i, 0]])


'''
Complete missing values in the data matrix
'''


def __complete_matrix(X):
    # get col means
    col_mean = np.nanmean(X, axis=0)
    # Find indicies that you need to replace
    idxs = np.where(np.isnan(X))
    X[idxs] = np.take(col_mean, idxs[1])


'''
Solution 1
'''

# Load data
X, y = __load_train_data()
X_test = __load_test_data()

# Do data processing
# 1. Replace NaNs by the colum means
__complete_matrix(X)
__complete_matrix(X_test)

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
__produce_solution(prediction_test)