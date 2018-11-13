import numpy as np
import sys
from sklearn import svm
from sklearn import model_selection
from biosppy.signals import ecg
from biosppy.signals import tools
from sklearn import ensemble
from sklearn import tree
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import ClusterCentroids
from pywt import wavedec

assert (len(sys.argv) > 1)


# def select_features_wavelet(X, SAMPLE_RADIUS=100, N_PEAKS=1):
#     ecg_res = ecg.ecg(X[0, :], 300, show=False)
#     x_filtered = ecg_res['filtered']
#     r_peak = ecg_res['rpeaks']
#
#     # samples
#     filtered_x = x_filtered[r_peak[3] - (SAMPLE_RADIUS - 1):r_peak[3] + (SAMPLE_RADIUS + 1)]
#     # wavelet
#     cA4, cD4, cD3, _, _ = wavedec(filtered_x, wavelet='sym6', level=4)
#     # n_features
#     FEATURE_LEN = np.array([len(cA4), len(cD4), len(cD3)])
#
#     # GET DIM
#     M = X.shape[0]
#     X_new = np.zeros((M,(FEATURE_LEN[0] + FEATURE_LEN[1] + FEATURE_LEN[2]) * N_PEAKS))
#
#     # Build the matrix
#     for i in range(M):
#         ecg_res = ecg.ecg(X[i], 300, False)
#         filtered_x = ecg_res['filtered']
#         r_peak = ecg_res['rpeaks']
#
#         cA4s = np.zeros((N_PEAKS, FEATURE_LEN[0]))
#         cD4s = np.zeros((N_PEAKS, FEATURE_LEN[1]))
#         cD3s = np.zeros((N_PEAKS, FEATURE_LEN[2]))
#
#         for j in range(1, N_PEAKS + 1):
#             new_feature = filtered_x[r_peak[j] - (SAMPLE_RADIUS-1):r_peak[j] + (SAMPLE_RADIUS+1)]
#             new_feature = tools.normalize(new_feature)
#
#             # wavelet
#             cA4, cD4, cD3, _, _ = wavedec(new_feature, wavelet='sym6', level=4)
#
#             cA4s[j-1, :] = cA4
#             cD4s[j-1, :] = cD4
#             cD3s[j-1, :] = cD3
#
#         # new features
#         X_new[i, :] = np.concatenate((cA4s.flatten(), cD4s.flatten(), cD3s.flatten()))
#
#     return X_new

# def select_features_sampling(X, SAMPLE_RADIUS=100):
#     FEATURE_LEN = SAMPLE_RADIUS * 2
#     M = X.shape[0]
#     X_new = np.zeros((M, FEATURE_LEN))
#     for i in range(M):
#         ecg_res = ecg.ecg(X[i], 300, False)
#         x_filtered = ecg_res['filtered']
#         r_peak = ecg_res['rpeaks']
#         new_feature = x_filtered[r_peak[1] - (SAMPLE_RADIUS - 1):r_peak[1] + (SAMPLE_RADIUS + 1)]
#         new_feature = tools.normalize(new_feature)
#         X_new[i] = np.reshape(new_feature, (1, FEATURE_LEN))
#     return X_new


# def select_features_beats(X, BEAT_LEN=50):
#     M = X.shape[0]
#     X_new = np.zeros((M, BEAT_LEN))
#     for i in range(M):
#         ecg_res = ecg.ecg(X[i], 300, False)
#         rate = ecg_res['heart_rate']
#         if rate.size == 0:
#             new_feature = np.zeros(BEAT_LEN)
#         elif rate.size < BEAT_LEN:
#             pad_len = round((BEAT_LEN - rate.size) / 2.00 + 0.001)
#             mode = 'symmetric'
#             new_feature = np.pad(rate, pad_len, mode)[0:BEAT_LEN]
#         elif rate.size > BEAT_LEN:
#             new_feature = rate[0:BEAT_LEN]
#         X_new[i] = new_feature
#     return X_new

def select_features_combo(X, BEAT_LEN=50, SAMPLE_RADIUS=100):
    M = X.shape[0]
    N = X.shape[1]
    NUM_INTERVAL = 20
    NUM_BINS_HISTO = 20
    X_new = np.zeros((M, BEAT_LEN + NUM_INTERVAL * 2 + NUM_INTERVAL * NUM_BINS_HISTO))
    for i in range(M):
        ecg_res = ecg.ecg(X[i], 300, False)
        rate = ecg_res['heart_rate']
        x_filtered = ecg_res['filtered']
        r_peak = ecg_res['rpeaks']

        # FEATURE 1
        if rate.size == 0:
            rate_feature = np.zeros(BEAT_LEN)
        elif rate.size < BEAT_LEN:
            pad_len = round((BEAT_LEN - rate.size) / 2.00 + 0.001)
            rate_feature = np.pad(rate, pad_len, 'symmetric')[0:BEAT_LEN]
        elif rate.size > BEAT_LEN:
            rate_feature = rate[0:BEAT_LEN]

        rate_feature = np.reshape(rate_feature, (-1, 1))

        # FEATURE 2
        idx_space = np.linspace(0, N, NUM_INTERVAL+1)
        mean_vec = []
        std_vec = []
        histogram_feature = np.array([])
        for j in range(1, len(idx_space)):
            interval = x_filtered[int(idx_space[j - 1]):int(idx_space[j])]
            mean_interval = np.mean(interval)
            std_interval = np.std(interval)
            mean_vec.append(mean_interval)
            std_vec.append(std_interval)
            hist, _ = np.histogram(interval, bins=NUM_BINS_HISTO)
            histogram_feature = np.append(histogram_feature, hist)

        mean_feature = np.array(mean_vec)
        std_feature = np.array(std_vec)

        # prev_idx = 0
        # for peak_idx in r_peak:
        #     RR_interval = x_filtered[prev_idx:peak_idx]
        #     prev_idx = peak_idx
        #     RR_histogram, _ = np.histogram(RR_interval, bins=20)

        rate_feature = rate_feature.ravel()
        new_feature = np.array([])
        new_feature = np.append(new_feature, rate_feature)
        new_feature = np.append(new_feature, histogram_feature)
        new_feature = np.append(new_feature, mean_feature)
        new_feature = np.append(new_feature, std_feature)

        X_new[i] = new_feature

    return X_new

# Read data
# print('Creating npy objects...')
# COL_NUM = 18154
# X_test = pd.read_csv('X_test.csv',sep=',',engine='python',header=0).values
# X_train = pd.read_csv('X_train.csv',sep=',',engine='python',header=0).values
# Y_train = np.genfromtxt('y_train.csv', delimiter=',', skip_header=1)
#
# np.save('X_train', X_train)
# np.save('Y_train', Y_train)
# np.save('X_test', X_test)

# Read data
print('Loading data...\n')
X_train = np.load('X_train.npy')[:, 1:]
Y_train = np.load('Y_train.npy').astype(int)[:, 1]
X_test = np.load('X_test.npy')
X_test_ids = X_test[:, 0]
X_test = X_test[:, 1:]

print('Heart beats TRAIN...\n')
X_train[np.isnan(X_train)] = 0
X_train = select_features_combo(X_train)
np.save('X_train_combo', X_train)
X_train = np.load('X_train_combo.npy')

# print('Heart beats TEST...\n')
# X_test[np.isnan(X_test)] = 0
# X_test = select_features(X_test)
# np.save('X_test_beats', X_test)
# X_test = np.load('X_test_beats.npy')

print('Cross-validating...\n')
clf = ensemble.RandomForestClassifier(n_estimators=200, max_depth=2, random_state=0)

clf_scores = model_selection.cross_val_score(clf, X_train, Y_train, cv=10, scoring='f1_micro')
print("\nCalculating the score")
print("N Scores are = {}".format(clf_scores))
print("Averaged F1 = {}".format(np.mean(clf_scores)))
print("STD of N scores = {}".format(np.std(clf_scores)))
clf.fit(X_train, Y_train.ravel())

# print('\nPredicting...')
# Y_test = clf.predict(X_test)
#
# np.savetxt("out_%s.csv" % sys.argv[1], np.stack((X_test_ids, Y_test), axis=1), delimiter=",", header="id,y",
#            fmt='%d', comments='')
