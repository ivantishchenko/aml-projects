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

X_train = util.get_band_features(train_eeg1)
y_train = train_labels.flatten()
X_test = util.get_band_features(test_eeg1)

X_train, mean_X, std_X = DataReader.normalize_data(X_train)
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


