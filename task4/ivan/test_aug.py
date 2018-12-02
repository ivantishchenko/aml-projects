import os
import math
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tf_utils import input_fn_from_dataset, input_fn_frame_from_dataset, save_tf_record, \
    prob_positive_class_from_prediction
from utils import save_solution
from data_manage import sliding_training_data, flip, normalize_data, rotate
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

dir_path = os.path.dirname(os.path.realpath(__file__))
my_solution_file = os.path.join(dir_path, '../solution.csv')

x_train = np.load('../X_train.npy')
y_train = np.load('../Y_train.npy')
x_test = np.load('../X_test.npy')

# So... can either create a RNN and run over frames
# or somehow extend videos so they are all [212, 100, 100]
# and then run a 3D CNN on this...
# however, maybe it is better to detect heartbeats and
# get features from these and then compute averages and stds over
# heartbeats and use these for classification

# Could also just run detection on each individual frame and average
# probabilities... however, then motion data would be missing...
# But, could also do a sliding window approach...
# i.e. run 3D convolution on (e.g.) 6 previous frames and then
# 2D convolution afterwards...
# Average vote again.

# Could take inspiration from ResNet or VGG and just replace
# first few convolutions with 3D

def predict_videos(X, model, blocksize=4):
    predictions = []
    for i in range(X.shape[0]):
        video = X[i]
        blocks = []
        for j in range(video.shape[0] // blocksize - 1):
            startIndex = j * blocksize
            endIndex = (j + 1) * blocksize
            block = video[startIndex:endIndex, :, :]
            block = block.reshape((1, blocksize, 100, 100))
            blocks.append(block)
        blocks = np.asarray(blocks)
        pred = model.predict(blocks)
        average = pred.mean(axis=0)
        predictions.append(average)
    predictions = np.asarray(predictions)
    return predictions

# Split into training and testing data
num_training = math.floor(x_train.shape[0] * 0.9)
indices = np.random.permutation(x_train.shape[0])
training_idx, validation_idx = indices[:num_training], indices[num_training:]
training_x = x_train[training_idx]
validation_x = x_train[validation_idx]
training_y = y_train[training_idx]
validation_y = y_train[validation_idx]


# Training data augmentation
training_x, training_y = rotate(training_x, training_y)
training_x, training_y = flip(training_x, training_y, horizontal=False, frames=True)

training_x = normalize_data(training_x)
validation_x = normalize_data(validation_x)
x_test = normalize_data(x_test)