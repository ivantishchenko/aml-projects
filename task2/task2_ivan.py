import numpy as np
import sklearn
import sklearn.model_selection
import sklearn.linear_model
import sklearn.ensemble
import sklearn.decomposition
import sklearn.svm
import sklearn.metrics
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
import sklearn.ensemble
import sklearn.gaussian_process
from sklearn import feature_selection
from sklearn import ensemble
import sys
from imblearn.combine import SMOTETomek
from sklearn import tree
from sklearn.gaussian_process.kernels import RBF
from sklearn import preprocessing
from sklearn import neighbors
from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import ClusterCentroids
from imblearn.over_sampling import SMOTE

'''
Solution 1
'''


def balanced_accuracy_score(y_true, y_pred, sample_weight=None,
                            adjusted=False):
    """Compute the balanced accuracy
    The balanced accuracy in binary and multiclass classification problems to
    deal with imbalanced datasets. It is defined as the average of recall
    obtained on each class.
    The best value is 1 and the worst value is 0 when ``adjusted=False``.
    Read more in the :ref:`User Guide <balanced_accuracy_score>`.
    Parameters
    ----------
    y_true : 1d array-like
        Ground truth (correct) target values.
    y_pred : 1d array-like
        Estimated targets as returned by a classifier.
    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.
    adjusted : bool, default=False
        When true, the result is adjusted for chance, so that random
        performance would score 0, and perfect performance scores 1.
    Returns
    -------
    balanced_accuracy : float
    See also
    --------
    recall_score, roc_auc_score
    Notes
    -----
    Some literature promotes alternative definitions of balanced accuracy. Our
    definition is equivalent to :func:`accuracy_score` with class-balanced
    sample weights, and shares desirable properties with the binary case.
    See the :ref:`User Guide <balanced_accuracy_score>`.
    References
    ----------
    .. [1] Brodersen, K.H.; Ong, C.S.; Stephan, K.E.; Buhmann, J.M. (2010).
           The balanced accuracy and its posterior distribution.
           Proceedings of the 20th International Conference on Pattern
           Recognition, 3121-24.
    .. [2] John. D. Kelleher, Brian Mac Namee, Aoife D'Arcy, (2015).
           `Fundamentals of Machine Learning for Predictive Data Analytics:
           Algorithms, Worked Examples, and Case Studies
           <https://mitpress.mit.edu/books/fundamentals-machine-learning-predictive-data-analytics>`_.
    Examples
    --------
    >>> from sklearn.metrics import balanced_accuracy_score
    >>> y_true = [0, 1, 0, 0, 1, 0]
    >>> y_pred = [0, 1, 0, 0, 0, 1]
    >>> balanced_accuracy_score(y_true, y_pred)
    0.625
    """
    C = sklearn.metrics.confusion_matrix(y_true, y_pred, sample_weight=sample_weight)
    with np.errstate(divide='ignore', invalid='ignore'):
        per_class = np.diag(C) / C.sum(axis=1)
    if np.any(np.isnan(per_class)):
        # warnings.warn('y_pred contains classes not in y_true')
        print('y_pred contains classes not in y_true')
        per_class = per_class[~np.isnan(per_class)]
    score = np.mean(per_class)
    if adjusted:
        n_classes = len(per_class)
        chance = 1 / n_classes
        score -= chance
        score /= 1 - chance
    return score


assert (len(sys.argv) > 1)

# Read data
print('Loading data...')
X_train = np.load('X_train.npy')[:, 1:]
Y_train = np.load('Y_train.npy').astype(int)[:, 1]
X_test = np.load('X_test.npy')
X_test_ids = X_test[:, 0]
X_test = X_test[:, 1:]

print('\nPrepocessing the dataset')
tomek = TomekLinks(return_indices=False, ratio='majority')
X_train, Y_train = tomek.fit_sample(X_train, Y_train.ravel())

# count the number of classes
unique, counts = np.unique(Y_train, return_counts=True)
clas_dict = dict(zip(unique, counts))
print('\nNumber of classes = {}'.format(clas_dict))

# scaler = preprocessing.StandardScaler().fit(X_train)
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)

# 3. Outlier detection
# lof = neighbors.LocalOutlierFactor(n_neighbors=60, contamination=0.005)
# outliers = lof.fit_predict(X_train)
#
# unique, counts = np.unique(outliers, return_counts=True)
# count_dict = dict(zip(unique, counts))
# X_train = X_train[outliers == 1]
# Y_train = Y_train[outliers == 1]

# select = feature_selection.SelectFromModel(ensemble.RandomForestRegressor(n_estimators=200, random_state=42))
# select.fit(X_train, Y_train)
# X_train = select.transform(X_train)
# X_test = select.transform(X_test)

# sm = SMOTETomek(ratio='auto')
# X_train, Y_train = sm.fit_sample(X_train, Y_train.ravel())

print('\nCross-validating...')
# SVC
# clf = sklearn.svm.SVC()
# clf = sklearn.model_selection.GridSearchCV(clf, {
#     'C': [1],
#     'kernel': ('rbf',),
#     'class_weight': ('balanced', None)
# }, scoring=sklearn.metrics.make_scorer(balanced_accuracy_score), cv=3, iid=False, verbose=2)
#
# clf.fit(X_train, Y_train.ravel())
#
# print('\nCV Results:')
# print(clf.cv_results_)
# print('\nBest parameters are:')
# print(clf.best_params_)

clf = sklearn.svm.SVC(kernel='rbf', class_weight='balanced')
clf_scores = cross_val_score(clf, X_train, Y_train, cv=10, scoring=sklearn.metrics.make_scorer(balanced_accuracy_score))
print("\nCalculating the score")
print("N Scores are = {}".format(clf_scores))
print("Averaged BMCA = {}".format(np.mean(clf_scores)))
print("STD of N scores = {}".format(np.std(clf_scores)))
clf.fit(X_train, Y_train.ravel())

print('\nPredicting...')
Y_test = clf.predict(X_test)

np.savetxt("out_%s.csv" % sys.argv[1], np.stack((X_test_ids, Y_test), axis=1), delimiter=",", header="id,y",
           fmt='%d', comments='')
