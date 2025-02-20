import numpy as np
import sys
from sklearn import model_selection
from biosppy.signals import ecg
from sklearn import ensemble
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
import time

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

# def select_features_combo(X, BEAT_LEN=50, SAMPLE_RADIUS=100):
#     M = X.shape[0]
#     N = X.shape[1]
#     NUM_INTERVAL = 20
#     NUM_BINS_HISTO = 20
#     X_new = np.zeros((M, BEAT_LEN + NUM_INTERVAL * 2 + NUM_INTERVAL * NUM_BINS_HISTO))
#     for i in range(M):
#         ecg_res = ecg.ecg(X[i], 300, False)
#         rate = ecg_res['heart_rate']
#         x_filtered = ecg_res['filtered']
#         r_peak = ecg_res['rpeaks']
#
#         # FEATURE 1
#         if rate.size == 0:
#             rate_feature = np.zeros(BEAT_LEN)
#         elif rate.size < BEAT_LEN:
#             pad_len = round((BEAT_LEN - rate.size) / 2.00 + 0.001)
#             rate_feature = np.pad(rate, pad_len, 'symmetric')[0:BEAT_LEN]
#         elif rate.size > BEAT_LEN:
#             rate_feature = rate[0:BEAT_LEN]
#
#         rate_feature = np.reshape(rate_feature, (-1, 1))
#
#         # FEATURE 2
#         idx_space = np.linspace(0, N, NUM_INTERVAL+1)
#         mean_vec = []
#         std_vec = []
#         histogram_feature = np.array([])
#         for j in range(1, len(idx_space)):
#             interval = x_filtered[int(idx_space[j - 1]):int(idx_space[j])]
#             mean_interval = np.mean(interval)
#             std_interval = np.std(interval)
#             mean_vec.append(mean_interval)
#             std_vec.append(std_interval)
#             hist, _ = np.histogram(interval, bins=NUM_BINS_HISTO)
#             histogram_feature = np.append(histogram_feature, hist)
#
#         mean_feature = np.array(mean_vec)
#         std_feature = np.array(std_vec)
#
#         # prev_idx = 0
#         # for peak_idx in r_peak:
#         #     RR_interval = x_filtered[prev_idx:peak_idx]
#         #     prev_idx = peak_idx
#         #     RR_histogram, _ = np.histogram(RR_interval, bins=20)
#
#         rate_feature = rate_feature.ravel()
#         new_feature = np.array([])
#         new_feature = np.append(new_feature, rate_feature)
#         new_feature = np.append(new_feature, histogram_feature)
#         new_feature = np.append(new_feature, mean_feature)
#         new_feature = np.append(new_feature, std_feature)
#
#         X_new[i] = new_feature
#
#     return X_new

# def select_features_combo(X, BEAT_LEN=50, SAMPLE_RADIUS=100):
#     M = X.shape[0]
#     N = X.shape[1]
#     NUM_INTERVAL = 20
#     NUM_BINS_HISTO = 20
#     SPLIT_T = 40
#
#     X_new = []
#     idx_template = np.linspace(0, 180, SPLIT_T+1)
#     for i in range(M):
#         new_feature = []
#         ecg_res = ecg.ecg(X[i], 300, False)
#         template = ecg_res['templates']
#         r_peak = ecg_res['rpeaks']
#
#         mean_vec = np.mean(template, axis=0)
#         var_vec = np.var(template, axis=0)
#         new_feature.extend(mean_vec)
#         new_feature.extend(var_vec)
#
#         for i in range(1, len(idx_template)):
#             data = np.ravel(template[:, int(idx_template[i - 1]):int(idx_template[i])])
#             hist, _ = np.histogram(data, bins=NUM_BINS_HISTO, density=True)
#             new_feature.extend(hist)
#
#         rr_diff = np.diff(r_peak)
#         if len(rr_diff) > 0:
#             hist, _ = np.histogram(rr_diff, bins=NUM_BINS_HISTO, density=True)
#             new_feature.extend(hist)
#         else:
#             new_feature.extend(np.zeros(NUM_BINS_HISTO))
#
#         X_new.append(new_feature)
#
#     X_new = np.array(X_new)
#     print(X_new.shape)
#     return X_new

def select_features_combo(X, BEAT_LEN=50, SAMPLE_RADIUS=100):
    M = X.shape[0]
    N = X.shape[1]
    NUM_SPLIT = 20
    NUM_BINS_HISTO = 20
    signal_space = np.linspace(0, N, NUM_SPLIT + 1)

    X_new = []
    for i in range(M):
        new_feature = []
        ecg_res = ecg.ecg(X[i], 300, False)
        templates = ecg_res['templates']
        rate = ecg_res['heart_rate']
        x_filtered = ecg_res['filtered']
        r_peaks = ecg_res['rpeaks']

        # FEATURE 1
        if rate.size == 0:
            rate_feature = np.zeros(BEAT_LEN)
        elif rate.size < BEAT_LEN:
            pad_len = round((BEAT_LEN - rate.size) / 2.00 + 0.001)
            rate_feature = np.pad(rate, pad_len, 'symmetric')[0:BEAT_LEN]
        elif rate.size > BEAT_LEN:
            rate_feature = rate[0:BEAT_LEN]
        rate_feature = np.reshape(rate_feature, (-1,))
        new_feature.extend(rate_feature)

        # FEATURE 2
        new_feature.extend(np.mean(templates, axis=0))
        new_feature.extend(np.var(templates, axis=0))

        # FEATURE 3
        rr_diff = np.diff(r_peaks)
        if len(rr_diff) > 0:
            hist, _ = np.histogram(rr_diff, bins=NUM_BINS_HISTO)
            new_feature.extend(hist)
        else:
            new_feature.extend(np.zeros(NUM_BINS_HISTO))

        # FEATURE 4
        r_peaks_christov = ecg.christov_segmenter(x_filtered, sampling_rate=300)['rpeaks']
        r_peaks_engzee = ecg.engzee_segmenter(x_filtered, sampling_rate=300)['rpeaks']
        r_peaks_hamilton = ecg.hamilton_segmenter(x_filtered, sampling_rate=300)['rpeaks']

        rr_diff_christov = np.diff(r_peaks_christov)
        rr_diff_engzee = np.diff(r_peaks_engzee)
        rr_diff_hamilton = np.diff(r_peaks_hamilton)
        if len(rr_diff_christov) > 0:
            hist, _ = np.histogram(rr_diff_christov, bins=NUM_BINS_HISTO)
            new_feature.extend(hist)
        else:
            new_feature.extend(np.zeros(NUM_BINS_HISTO))

        if len(rr_diff_engzee) > 0:
            hist, _ = np.histogram(rr_diff_engzee, bins=NUM_BINS_HISTO)
            new_feature.extend(hist)
        else:
            new_feature.extend(np.zeros(NUM_BINS_HISTO))

        if len(rr_diff_hamilton) > 0:
            hist, _ = np.histogram(rr_diff_hamilton, bins=NUM_BINS_HISTO)
            new_feature.extend(hist)
        else:
            new_feature.extend(np.zeros(NUM_BINS_HISTO))

        # FEATURE 5
        R_amplitudes = x_filtered[r_peaks]
        hist, _ = np.histogram(R_amplitudes, bins=NUM_BINS_HISTO)
        new_feature.extend(hist)
        # All features
        X_new.append(new_feature)

        # FEATURE 6
        lower_idx1 = 0
        lower_idx2 = 0
        lower_idx3 = 0
        lower_idx4 = 0
        for j in range(1, len(signal_space)):
            if len(np.where(r_peaks < signal_space[j])[0]) > 0:
                upped_idx1 = max(np.where(r_peaks < signal_space[j])[0])
                if len(rr_diff[lower_idx1:upped_idx1 + 1]) > 0:
                    new_feature.append(np.mean(rr_diff[lower_idx1:upped_idx1 + 1]))
                    new_feature.append(np.var(rr_diff[lower_idx1:upped_idx1 + 1]))
                else:
                    new_feature.append(-1)
                    new_feature.append(-1)
                lower_idx1 = upped_idx1 + 1
            else:
                new_feature.append(-1)
                new_feature.append(-1)

            if len(np.where(r_peaks_christov < signal_space[j])[0]) > 0:
                upped_idx2 = max(np.where(r_peaks_christov < signal_space[j])[0])
                if len(rr_diff_christov[lower_idx2:upped_idx2 + 1]) > 0:
                    new_feature.append(np.mean(rr_diff_christov[lower_idx2:upped_idx2 + 1]))
                    new_feature.append(np.var(rr_diff_christov[lower_idx2:upped_idx2 + 1]))
                else:
                    new_feature.append(-1)
                    new_feature.append(-1)
                lower_idx2 = upped_idx2 + 1
            else:
                new_feature.append(-1)
                new_feature.append(-1)

            if len(np.where(r_peaks_engzee < signal_space[j])[0]) > 0:
                upped_idx3 = max(np.where(r_peaks_engzee < signal_space[j])[0])
                if len(rr_diff_engzee[lower_idx3:upped_idx3 + 1]) > 0:
                    new_feature.append(np.mean(rr_diff_engzee[lower_idx3:upped_idx3 + 1]))
                    new_feature.append(np.var(rr_diff_engzee[lower_idx3:upped_idx3 + 1]))
                else:
                    new_feature.append(-1)
                    new_feature.append(-1)
                lower_idx3 = upped_idx3 + 1
            else:
                new_feature.append(-1)
                new_feature.append(-1)

            if len(np.where(r_peaks_hamilton < signal_space[j])[0]) > 0:
                upped_idx4 = max(np.where(r_peaks_hamilton < signal_space[j])[0])
                if len(rr_diff_hamilton[lower_idx4:upped_idx4 + 1]) > 0:
                    new_feature.append(np.mean(rr_diff_hamilton[lower_idx4:upped_idx4 + 1]))
                    new_feature.append(np.var(rr_diff_hamilton[lower_idx4:upped_idx4 + 1]))
                else:
                    new_feature.append(-1)
                    new_feature.append(-1)
                lower_idx4 = upped_idx4 + 1
            else:
                new_feature.append(-1)
                new_feature.append(-1)

    X_new = np.array(X_new)
    print(X_new.shape)
    return X_new


'''
TRAINING
'''
start = time.time()

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
# X_train[np.isnan(X_train)] = 0
# X_train = select_features_combo(X_train)
# np.save('X_train_combo', X_train)
X_train = np.load('X_train_combo.npy')

print('Cross-validating...\n')
clf = ensemble.RandomForestClassifier(n_estimators=1000, max_depth=30, random_state=0, n_jobs=-1)

clf_scores = model_selection.cross_val_score(clf, X_train, Y_train, cv=10, scoring='f1_micro')
print("\nCalculating the score")
print("N Scores are = {}".format(clf_scores))
print("Averaged F1 = {}".format(np.mean(clf_scores)))
print("STD of N scores = {}".format(np.std(clf_scores)))
clf.fit(X_train, Y_train.ravel())

print('Heart beats TEST...\n')
# X_test[np.isnan(X_test)] = 0
# X_test = select_features_combo(X_test)
# np.save('X_test_combo', X_test)
X_test = np.load('X_test_combo.npy')

print('\nPredicting...')
Y_test = clf.predict(X_test)

np.savetxt("out_%s.csv" % sys.argv[1], np.stack((X_test_ids, Y_test), axis=1), delimiter=",", header="id,y",
           fmt='%d', comments='')

end = time.time()
print(end - start)
