import csv

import numpy as np

TRAIN_M = 1212
TRAIN_N = 887

TEST_M = 776
TEST_N = 887


def load_test_data():
    """
    Load the test data
    """

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


def load_train_data():
    """
    Load the train data
    """

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


def produce_solution(y):
    """
    Produce the CSV of a solution
    """

    with open('out.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', lineterminator="\n")
        writer.writerow(['id', 'y'])
        for i in range(y.shape[0]):
            writer.writerow([float(i), y[i, 0]])
