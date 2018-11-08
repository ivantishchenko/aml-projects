import numpy as np
import sys
from sklearn import svm
from sklearn import model_selection
from biosppy.signals import ecg
from sklearn.ensemble import RandomForestClassifier

assert (len(sys.argv) > 1)


def select_features(X, BEAT_LEN=50):
    M = X.shape[0]
    X_new = np.zeros((M, BEAT_LEN))
    for i in range(M):
        rate = ecg.ecg(X[i], 300, False)['heart_rate']
        if rate.size == 0:
            new_feature = np.zeros(BEAT_LEN)
        elif rate.size < BEAT_LEN:
            pad_len = round((BEAT_LEN - rate.size) / 2.00 + 0.001)
            mode = 'symmetric'
            new_feature = np.pad(rate, pad_len, mode)[0:BEAT_LEN]
        elif rate.size > BEAT_LEN:
            new_feature = rate[0:BEAT_LEN]
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
# X_train[np.isnan(X_train)] = 0
# X_train = select_features(X_train)
# np.save('X_train_beats', X_train)
X_train = np.load('X_train_beats.npy')

print('Heart beats TEST...\n')
# X_test[np.isnan(X_test)] = 0
# X_test = select_features(X_test)
# np.save('X_test_beats', X_test)
X_test = np.load('X_test_beats.npy')

print('Cross-validating...\n')
clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
# clf = svm.SVC(kernel='rbf', class_weight='balanced')

clf_scores = model_selection.cross_val_score(clf, X_train, Y_train, cv=10, scoring='f1_micro')
print("\nCalculating the score")
print("N Scores are = {}".format(clf_scores))
print("Averaged F1 = {}".format(np.mean(clf_scores)))
print("STD of N scores = {}".format(np.std(clf_scores)))
clf.fit(X_train, Y_train.ravel())

print('\nPredicting...')
Y_test = clf.predict(X_test)

np.savetxt("out_%s.csv" % sys.argv[1], np.stack((X_test_ids, Y_test), axis=1), delimiter=",", header="id,y",
           fmt='%d', comments='')
