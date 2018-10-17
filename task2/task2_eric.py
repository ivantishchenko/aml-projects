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
import sys

assert(len(sys.argv) > 1)

# Read data
print('Loading data...')

# X_test  = np.genfromtxt('X_test.csv', delimiter=',', skip_header=1)
# X_train = np.genfromtxt('X_train.csv', delimiter=',', skip_header=1)
# Y_train = np.genfromtxt('y_train.csv', delimiter=',', skip_header=1)
# np.save('X_train', X_train)
# np.save('Y_train', Y_train)
# np.save('X_test', X_test)

X_train = np.load('X_train.npy')[:, 1:]
Y_train = np.load('Y_train.npy').astype(int)[:, 1]
X_test = np.load('X_test.npy')
X_test_ids = X_test[:, 0]
X_test = X_test[:, 1:]


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
        warnings.warn('y_pred contains classes not in y_true')
        per_class = per_class[~np.isnan(per_class)]
    score = np.mean(per_class)
    if adjusted:
        n_classes = len(per_class)
        chance = 1 / n_classes
        score -= chance
        score /= 1 - chance
    return score


print('Cross-validating...')

# # KNN
# print('KNN')
# clf = make_pipeline(sklearn.preprocessing.Normalizer(), sklearn.neighbors.KNeighborsClassifier()) 
# clf = sklearn.model_selection.GridSearchCV(clf, {
# 	'kneighborsclassifier__weights': ('uniform', 'distance'),
# 	'kneighborsclassifier__n_neighbors': [3,5,10],
# 	'kneighborsclassifier__metric': ('euclidean',)
# }, scoring=sklearn.metrics.make_scorer(balanced_accuracy_score), cv=3, iid=False, verbose=1, n_jobs=16)
# clf.fit(X_train, Y_train.ravel())
# print(clf.cv_results_)
# print(clf.best_params_)

# # # RF
# print('\nRF')
# clf = sklearn.ensemble.RandomForestClassifier()
# clf = sklearn.model_selection.GridSearchCV(clf, {
# 	'n_estimators': [100, 150, 200],
# 	'criterion': ('gini', 'entropy'),
# 	'class_weight': (None, 'balanced', 'balanced_subsample')
# }, scoring=sklearn.metrics.make_scorer(balanced_accuracy_score), cv=3, iid=False, verbose=1, n_jobs=16)
# clf.fit(X_train, Y_train.ravel())
# print(clf.cv_results_)
# print(clf.best_params_)


# SVC
print('\nSVC')
clf = sklearn.svm.SVC()
clf = sklearn.model_selection.GridSearchCV(clf, {
	'C': [1],
    'kernel': ('rbf',)
	'class_weight': ('balanced',None)
}, scoring=sklearn.metrics.make_scorer(balanced_accuracy_score), cv=3, iid=False, verbose=1, n_jobs=16)
clf.fit(X_train, Y_train.ravel())
print(clf.cv_results_)
print(clf.best_params_)




# score = cross_val_score(clf, X_train, Y_train.ravel(), cv=3, scoring=sklearn.metrics.make_scorer(balanced_accuracy_score))
# print(score)

print('Predicting...')
#clf.fit(X_train, Y_train)
Y_test = clf.predict(X_test)

np.savetxt("Y_test_%s.csv" % sys.argv[1], np.stack(( X_test_ids, Y_test ), axis=1), 
	delimiter=",", header="id,y", fmt='%d', comments='')
