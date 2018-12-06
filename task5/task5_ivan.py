import numpy as np
import sys

assert(len(sys.argv) > 1)

# Read data
print('Loading data...')
# EEG1_train = np.genfromtxt('train_eeg1.csv', delimiter=',', skip_header=1)
# EEG2_train = np.genfromtxt('train_eeg2.csv', delimiter=',', skip_header=1)
# EMG_train = np.genfromtxt('train_emg.csv', delimiter=',', skip_header=1)

# Y_train = np.genfromtxt('train_labels.csv', delimiter=',', skip_header=1)

# EEG1_test = np.genfromtxt('test_eeg1.csv', delimiter=',', skip_header=1)
# EEG2_test = np.genfromtxt('test_eeg2.csv', delimiter=',', skip_header=1)
# EMG_test = np.genfromtxt('test_emg.csv', delimiter=',', skip_header=1)
#
# np.save('EEG1_train', EEG1_train)
# np.save('EEG2_train', EEG2_train)
# np.save('EMG_train', EMG_train)
#
# np.save('Y_train', Y_train)
#
# np.save('EEG1_test', EEG1_test)
# np.save('EEG2_test', EEG2_test)
# np.save('EMG_test', EMG_test)

EEG1_train = np.load('EEG1_train.npy')[:, 1:]
EEG2_train = np.load('EEG2_train.npy')[:, 1:]
EMG_train = np.load('EMG_train.npy')[:, 1:]
Y_train = np.load('Y_train.npy').astype(int)[:, 1]

EEG1_test = np.load('EEG1_test.npy')[:, 1:]
EEG2_test = np.load('EEG2_test.npy')[:, 1:]
EMG_test = np.load('EMG_test.npy')
test_ids = EMG_test[:, 0]
EMG_test = EMG_test[:, 1:]

pass