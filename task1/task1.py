import numpy as np

TRAIN_M = 1212
TRAIN_N = 887

X_train = np.array((TRAIN_M, TRAIN_N))
y_train = np.array((TRAIN_M, 1))

def __load_train_data():
    with open('test.csv') as csvfile:
        reader = csv.DictReader(csvfile)