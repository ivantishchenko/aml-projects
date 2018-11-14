import numpy as np
import pandas as pd

# Read data
print('Creating npy objects...')
COL_NUM = 18154
X_test = pd.read_csv('X_test.csv',sep=',',engine='python',header=0).values
X_train = pd.read_csv('X_train.csv',sep=',',engine='python',header=0).values
Y_train = np.genfromtxt('y_train.csv', delimiter=',', skip_header=1)

np.save('X_train', X_train)
np.save('Y_train', Y_train)
np.save('X_test', X_test)