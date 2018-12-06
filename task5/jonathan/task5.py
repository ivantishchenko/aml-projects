import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import biosppy
import util
import DataReader

train_eeg1, train_eeg1_id = DataReader.read_data("../train_eeg1.csv")
train_eeg2, train_eeg2_id = DataReader.read_data("../train_eeg2.csv")
train_emg, train_emg_id = DataReader.read_data("../train_emg.csv")
test_eeg1, test_eeg1_id = DataReader.read_data("../test_eeg1.csv")
test_eeg2, test_eeg2_id = DataReader.read_data("../test_eeg2.csv")
test_emg, test_emg_id = DataReader.read_data("../test_emg.csv")
train_labels, train_id = DataReader.read_data("../train_labels.csv")

print(train_eeg1.shape)
print(train_eeg2.shape)
print(train_emg.shape)
print(train_labels.shape)
print(test_eeg1_id.shape)

# Each row is 4x128 values (512) where 4 is number of seconds per "epoch"
# 128 is the measurement frequency

# Each subject has 21600 epochs (24 hours)
# Three subjects => Total training data: 64800 epochs
# Neighboring epochs are temporally coherent...

# Task: Classify each epoch as either 1 (wake), 2 (nrem), 3 (rem)
#eeg1_train_signal = util.create_eeg(train_eeg1)
#X_train = util.get_eeg_features(eeg1_train_signal)
#np.save("X_train.npy", X_train)
#np.save("y_train.npy", train_labels)
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy").flatten()

X_train, mean_X, std_X = DataReader.normalize_data(X_train)

eeg1_test_signal = util.create_eeg(test_eeg1)
X_test = util.get_eeg_features(eeg1_test_signal)
np.save("X_test.npy", X_test)
X_test = np.load("X_test.npy")
X_test, _, _ = DataReader.normalize_data(X_test, mean_X, std_X)

kf = KFold(n_splits=3)
clf = RandomForestClassifier(n_estimators=1000, class_weight="balanced")
i = 0

for train_index, validation_index in kf.split(X_train, y_train):
    training_x, validation_x = X_train[train_index, :], X_train[validation_index, :]
    training_y, validation_y = y_train[train_index], y_train[validation_index]
    clf.fit(training_x, training_y)
    pred_y = clf.predict(validation_x)
    score = balanced_accuracy_score(validation_y, pred_y)
    print(str(i) + ": " + str(score))
    if i == 0:
        subm_y = clf.predict(X_test)
    else:
        subm_y = subm_y + clf.predict(X_test)
    i = i + 1
subm_y = subm_y / i

print(subm_y)

f = open("submission.csv", "w")
f.write("Id,y\n")
for i in range(test_eeg1_id.shape[0]):
    f.write(str(int(test_eeg1_id[i])) + "," + str(round(subm_y[i])) + "\n")
f.close()



