from sklearn.feature_selection import SelectPercentile
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn import preprocessing
from sklearn import neighbors
from sklearn import feature_selection
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn import ensemble
from sklearn import neural_network
from sklearn import svm
from sklearn import kernel_ridge
from sklearn.decomposition import KernelPCA
from sklearn.feature_selection import RFECV
# Constants
import csv
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel

# In[79]:


from sklearn.model_selection import KFold


def produce_solution(y):
    """
    Produce the CSV of a solution
    """

    with open('notOverfitted.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', lineterminator="\n")
        writer.writerow(['id', 'y'])
        for i in range(y.shape[0]):
            writer.writerow([float(i), y[i]])


def to_be_eliminated(df):
    # Create correlation matrix
    corr_matrix = df.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Find features with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

    return to_drop


def load_data():
    X_train = pd.read_csv("X_train.csv")
    X_test = pd.read_csv("X_test.csv")
    y_train = pd.read_csv("y_train.csv")

    X_train = X_train.drop('id', axis=1)
    X_test = X_test.drop('id', axis=1)
    y_train = y_train.drop('id', axis=1)

    to_drop = to_be_eliminated(X_train)

    for i in range(len(to_drop)):
        X_train = X_train.drop(to_drop[i], axis=1)

    for i in range(len(to_drop)):
        X_test = X_test.drop(to_drop[i], axis=1)
    # print(len(X_test.columns))
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    return X_train, X_test, y_train


def __complete_matrix_colmean(X):
    """
    Complete missing values in the data matrix
    """

    # get col means
    col_mean = np.nanmean(X, axis=0)
    # Find indicies that you need to replace
    idxs = np.where(np.isnan(X))
    X[idxs] = np.take(col_mean, idxs[1])


def __complete_matrix_zeros(X):
    """
    Completion with zeros
    """

    idxs = np.isnan(X)
    X[idxs] = 0


def low_rank_approx(A=None, r=1):
    """
    Computes an r-rank approximation of a matrix
    given the component u, s, and v of it's SVD
    Requires: numpy
    """
    SVD = np.linalg.svd(A)
    u, s, v = SVD
    Ar = np.zeros((len(u), len(v)))
    for i in range(r):
        Ar += s[i] * np.outer(u.T[i], v[i])
    return Ar


'''
Solution 1
'''


###### best crossval procedure ###
from sklearn.metrics import r2_score
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

X, X_Test, y = load_data()
kf = KFold(n_splits=10)
scores = np.array([])

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # filling the nans

    print("Complete training and testing")
    __complete_matrix_colmean(X_train)
    __complete_matrix_colmean(X_test)

    # 2. Zero Mean, Unit Variance
    print("Standardize data")
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # 3. Outlier detection
    print("Outlier Detection")
    lof = neighbors.LocalOutlierFactor(n_neighbors=10, contamination=0.005)
    outliers = lof.fit_predict(X_train)

    unique, counts = np.unique(outliers, return_counts=True)
    count_dict = dict(zip(unique, counts))
    X_train = X_train[outliers == 1]
    y_train = y_train[outliers == 1]

    # 4. Feature selection 
    print("Feature Selection")
    select = SelectFromModel(RandomForestRegressor(n_estimators=100, random_state=42))
    select.fit(X_train, y_train)
    X_train = select.transform(X_train)

    print("Fitting the model")
    # reg = ensemble.RandomForestRegressor(n_estimators=300)
    reg = AdaBoostRegressor(DecisionTreeRegressor(max_depth=50), n_estimators=1000, random_state=42)
    reg.fit(X_train, y_train)

    # prediction
    print("Predicting")
    X_test = select.transform(X_test)
    pred = reg.predict(X_test)

    # scoring

    score = r2_score(y_test, pred)
    # print(score)
    scores = np.append(scores, score)

# In[105]:


truth = np.mean(scores)
std = np.std(scores)
print("mean expected error: ", truth, "std: ", std)

# In[ ]:


X_test = select.transform(X_test)

# In[92]:


X, X_test, y = load_data()

__complete_matrix_colmean(X)
__complete_matrix_colmean(X_test)

scaler = preprocessing.StandardScaler().fit(X)
X = scaler.transform(X)
X_test = scaler.transform(X_test)

# 3. Outlier detection
# LocalOutlierFactor
lof = neighbors.LocalOutlierFactor(n_neighbors=60, contamination=0.005)
outliers = lof.fit_predict(X)

unique, counts = np.unique(outliers, return_counts=True)
count_dict = dict(zip(unique, counts))
X = X[outliers == 1]
y = y[outliers == 1]

# 4. Feature selection 

select = SelectFromModel(RandomForestRegressor(n_estimators=100, random_state=42))
select.fit(X, y)
X = select.transform(X)
X_test = select.transform(X_test)
print(X.shape)

reg = ensemble.RandomForestRegressor(n_estimators=200)
reg.fit(X, y)
pred = reg.predict(X_test)

# In[96]:


produce_solution(pred)
