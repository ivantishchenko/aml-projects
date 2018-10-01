import csv
import numpy as np
from sklearn import linear_model


TRAIN_M = 1212
TRAIN_N = 887

TEST_M = 776
TEST_N = 887

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

X_train, y_train = __load_train_data()
X_test = __load_test_data()

