import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
import util
import DataReader

save = False
if save:
    print('Loading data...')

    train_eeg1 = np.load('../EEG1_train.npy')[:, 1:]
    train_eeg2 = np.load('../EEG2_train.npy')[:, 1:]
    train_emg = np.load('../EMG_train.npy')[:, 1:]
    y_train = np.load('../Y_train.npy').astype(int)[:, 1]

    test_eeg1 = np.load('../EEG1_test.npy')[:, 1:]
    test_eeg2 = np.load('../EEG2_test.npy')[:, 1:]
    test_emg = np.load('../EMG_test.npy')
    test_eeg1_id = test_emg[:, 0]
    test_emg = test_emg[:, 1:]

    print('Reslicing...')
    tmp, y_train = util.reslice(np.stack((train_eeg1, train_eeg2, train_emg), axis=2), y_train)
    train_eeg1, train_eeg2, train_emg = tmp[:,:,0], tmp[:,:,1], tmp[:,:,2]
    np.save('Y_train3.npy', y_train)
    print('Resliced to %d samples' % len(train_eeg1))

    print('Calculating train features...')

    X_train_eeg3 = util.get_band_features(train_eeg1)
    X_train_eeg4 = util.get_band_features(train_eeg2)

    X_train_eeg1 = util.get_wavelet_coeffs(train_eeg1)
    X_train_eeg1 = util.get_wavelet_features(X_train_eeg1)
    X_train_eeg2 = util.get_wavelet_coeffs(train_eeg2)
    X_train_eeg2 = util.get_wavelet_features(X_train_eeg2)
    
    X_train_emg = util.get_emg_features(train_emg)
    X_train_emg2 = util.get_wavelet_coeffs(train_emg)
    X_train_emg2 = util.get_wavelet_features(X_train_emg2)

    X_train = np.concatenate((X_train_eeg1, X_train_eeg2, X_train_eeg3, X_train_eeg4, X_train_emg, X_train_emg2), axis=1)
    np.save("X_train3.npy", X_train)

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
    np.save("X_test3.npy", X_test)
    X_test = np.load("X_test3.npy")
else:
    X_train = np.load("X_train3.npy")
    X_test = np.load("X_test3.npy")
    y_train = np.load("Y_train3.npy")

# y_train = train_labels.flatten()
X_train, mean_X, std_X = DataReader.normalize_data(X_train)
X_test, _, _ = DataReader.normalize_data(X_test, mean_X, std_X)
X_train = DataReader.replace_infs(DataReader.replace_nans(X_train, 0.0), 0.0)
X_test = DataReader.replace_infs(DataReader.replace_nans(X_test, 0.0), 0.0)
print("Data prepared... Classifying...")

print(X_train.shape)
print(y_train.shape)
#selector = SelectKBest(k=50)
#X_train = selector.fit_transform(X_train, y_train)
#X_test = selector.transform(X_test)
print(X_train.shape)

#kf = KFold(n_splits=10)
clf = SVC(gamma='auto', class_weight='balanced')

# i = 0
# for train_index, validation_index in kf.split(X_train, y_train):
#     training_x, validation_x = X_train[train_index, :], X_train[validation_index, :]
#     training_y, validation_y = y_train[train_index], y_train[validation_index]
#     clf.fit(training_x, training_y)
#     pred_y = clf.predict(validation_x)
#     score = balanced_accuracy_score(validation_y, pred_y)
#     print(str(i) + ": " + str(score))
#     if i == 0:
#         subm_y = clf.predict(X_test)
#     else:
#         subm_y = subm_y + clf.predict(X_test)
#     i = i + 1
# subm_y = subm_y / i

# print(subm_y)

clf.fit(X_train, y_train)

print('Predicting...')
subm_y = clf.predict(X_test)

f = open("submission.csv", "w")
f.write("Id,y\n")
for i in range(X_test.shape[0]):
    f.write(str(i) + "," + str(round(subm_y[i])) + "\n")
f.close()

