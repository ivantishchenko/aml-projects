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
from sklearn.feature_selection import SelectKBest, SelectFdr
from sklearn.neighbors import KNeighborsClassifier
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

save = True
if save:
    X_train_eeg1 = util.get_wavelet_coeffs(train_eeg1)
    X_train_eeg1 = util.get_wavelet_features(X_train_eeg1)
    X_train_eeg2 = util.get_wavelet_coeffs(train_eeg2)
    X_train_eeg2 = util.get_wavelet_features(X_train_eeg2)
    
    X_train_eeg3 = util.get_band_features(train_eeg1)
    X_train_eeg4 = util.get_band_features(train_eeg2)
    X_train_emg = util.get_emg_features(train_emg)
    X_train_emg2 = util.get_wavelet_coeffs(train_emg)
    X_train_emg2 = util.get_wavelet_features(X_train_emg2)

    X_train = np.concatenate((X_train_eeg1, X_train_eeg2, X_train_eeg3, X_train_eeg4, X_train_emg, X_train_emg2), axis=1)

    X_test_eeg1 = util.get_wavelet_coeffs(test_eeg1)
    X_test_eeg1 = util.get_wavelet_features(X_test_eeg1)
    X_test_eeg2 = util.get_wavelet_coeffs(test_eeg2)
    X_test_eeg2 = util.get_wavelet_features(X_test_eeg2)
    X_test_eeg3 = util.get_band_features(test_eeg1)
    X_test_eeg4 = util.get_band_features(test_eeg2)
    X_test_emg = util.get_emg_features(test_emg)
    X_test_emg2 = util.get_wavelet_coeffs(test_emg)
    X_test_emg2 = util.get_wavelet_features(X_test_emg2)

    X_test = np.concatenate((X_test_eeg1, X_test_eeg2, X_test_eeg3, X_test_eeg4, X_test_emg, X_test_emg2), axis=1)

    np.save("X_train3.npy", X_train)
    np.save("X_test3.npy", X_test)
else:
    X_train = np.load("X_train3.npy")
    X_test = np.load("X_test3.npy")

y_train = train_labels.flatten()
X_train, mean_X, std_X = DataReader.normalize_data(X_train)
X_test, _, _ = DataReader.normalize_data(X_test, mean_X, std_X)
X_train = DataReader.replace_infs(DataReader.replace_nans(X_train, 0.0), 0.0)
X_test = DataReader.replace_infs(DataReader.replace_nans(X_test, 0.0), 0.0)
print("Data prepared... Classifying...")

kf = KFold(n_splits=10)
clf = SVC(gamma='auto', class_weight='balanced')
#clf = RandomForestClassifier(n_estimators=100, class_weight='balanced')
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

