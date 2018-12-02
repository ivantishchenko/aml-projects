import os
import math
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tf_utils import input_fn_from_dataset, input_fn_frame_from_dataset, save_tf_record, \
    prob_positive_class_from_prediction
from utils import save_solution
from data_manage import sliding_training_data, flip, normalize_data, extend_videos, get_frames
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

dir_path = os.path.dirname(os.path.realpath(__file__))
my_solution_file = os.path.join(dir_path, '../solution.csv')

x_train = np.load('../X_train.npy')
y_train = np.load('../Y_train.npy')
x_test = np.load('../X_test.npy')

# Make into -1 to 1 range
x_train = normalize_data(x_train)
x_test = normalize_data(x_test)

# Split into training and validation
num_training = math.floor(x_train.shape[0] * 0.9)
indices = np.random.permutation(x_train.shape[0])
training_idx, validation_idx = indices[:num_training], indices[num_training:]
training_x = x_train[training_idx]
validation_x = x_train[validation_idx]
training_y = y_train[training_idx]
validation_y = y_train[validation_idx]


def predict_videos(X, model):
    predictions = []
    for i in range(X.shape[0]):
        frames = X[i].reshape((X[i].shape[0], 100, 100, 1))
        pred = model.predict(frames)
        average = pred.mean(axis=0)
        predictions.append(average)
    predictions = np.asarray(predictions)
    return predictions


# Training data augmentation
training_x, training_y = flip(training_x, training_y, horizontal=False, frames=True)

splits = 10
kf = KFold(n_splits=splits)
frames, labels = get_frames(training_x, training_y)
print(frames.shape)
print(labels.shape)
models = []

for train_index, valid_index in kf.split(frames, labels):
    train_frames = frames[train_index, :, :]
    valid_frames = frames[valid_index, :, :]
    train_labels = labels[train_index]
    valid_labels = labels[valid_index]

    cnn = keras.Sequential()
    cnn.add(keras.layers.InputLayer(input_shape=(100, 100, 1)))

    cnn.add(keras.layers.Conv2D(32, 3, padding='same'))
    cnn.add(keras.layers.BatchNormalization())
    cnn.add(keras.layers.Activation('relu'))

    cnn.add(keras.layers.Conv2D(32, 3, padding='same'))
    cnn.add(keras.layers.BatchNormalization())
    cnn.add(keras.layers.Activation('relu'))

    cnn.add(keras.layers.MaxPooling2D(2))  # 50x50

    cnn.add(keras.layers.Conv2D(64, 3, padding='same'))
    cnn.add(keras.layers.BatchNormalization())
    cnn.add(keras.layers.Activation('relu'))

    cnn.add(keras.layers.Conv2D(64, 3, padding='same'))
    cnn.add(keras.layers.BatchNormalization())
    cnn.add(keras.layers.Activation('relu'))

    cnn.add(keras.layers.MaxPooling2D(2))  # 25x25

    cnn.add(keras.layers.Conv2D(128, 3, padding='same'))
    cnn.add(keras.layers.BatchNormalization())
    cnn.add(keras.layers.Activation('relu'))

    cnn.add(keras.layers.Conv2D(128, 3, padding='same'))
    cnn.add(keras.layers.BatchNormalization())
    cnn.add(keras.layers.Activation('relu'))

    cnn.add(keras.layers.MaxPooling2D(2))  # 12x12

    cnn.add(keras.layers.Flatten())
    cnn.add(keras.layers.Dense(1028, activation=tf.nn.relu))
    cnn.add(keras.layers.Dropout(0.25))
    cnn.add(keras.layers.Dense(512, activation=tf.nn.relu))
    cnn.add(keras.layers.Dropout(0.25))
    cnn.add(keras.layers.Dense(2, activation=tf.nn.softmax))

    cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    cnn.fit(train_frames, train_labels, epochs=10)
    models.append(cnn)

solutions = []
for i in range(splits):
    # Validation
    vpred = predict_videos(validation_x, models[i])
    probs_pos = [prob[1] for prob in vpred]
    probs_pos = np.asarray(probs_pos)
    roc_auc = roc_auc_score(validation_y, probs_pos)
    print(roc_auc)

    # Prediction
    predictions = predict_videos(x_test, models[i])
    solution = [prob[1] for prob in predictions]
    solutions.append(solution)

solutions = np.asarray(solutions)
solution = list(solutions.mean(axis=0))
save_solution(my_solution_file, solution)
