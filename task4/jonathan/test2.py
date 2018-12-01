import os
import math
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tf_utils import input_fn_from_dataset,input_fn_frame_from_dataset,save_tf_record,prob_positive_class_from_prediction
from get_data import get_videos_from_folder,get_target_from_csv
from utils import save_solution
from data_manage import sliding_training_data, flip, normalize_data, extend_videos
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

dir_path = os.path.dirname(os.path.realpath(__file__))
train_folder = os.path.join(dir_path,"../train/")
test_folder = os.path.join(dir_path,"../test/")

train_target = os.path.join(dir_path,'../train_target.csv')
my_solution_file = os.path.join(dir_path,'../solution.csv')

x_train = get_videos_from_folder(train_folder)
y_train = get_target_from_csv(train_target)
x_test = get_videos_from_folder(test_folder)

# Make into -1 to 1 range
x_train = normalize_data(x_train)
x_test = normalize_data(x_test)

# Extend videos so that they are all the same length (216, 100, 100)
x_train = extend_videos(x_train)
x_test = extend_videos(x_test)
x_train = x_train.reshape((x_train.shape[0], 1, x_train[0].shape[0], x_train[0].shape[1], x_train[0].shape[2]))
x_test = x_test.reshape((x_test.shape[0], 1, x_test[0].shape[0], x_test[0].shape[1], x_test[0].shape[2]))

print(x_train.shape)

# Split into training and validation
num_training = math.floor(x_train.shape[0] * 0.9)
indices = np.random.permutation(x_train.shape[0])
training_idx, validation_idx = indices[:num_training], indices[num_training:]
training_x = x_train[training_idx]
validation_x = x_train[validation_idx]
training_y = y_train[training_idx]
validation_y = y_train[validation_idx]

model = keras.Sequential([
			keras.layers.InputLayer(input_shape=(1, 216, 100, 100)),
			keras.layers.Conv3D(64, 3, strides=(2,2,2), activation=tf.nn.relu, padding='same'), # 108x50x50
			keras.layers.Conv3D(128, 3, strides=(2,2,2), activation=tf.nn.relu, padding='same'), # 54x25x25
			keras.layers.Flatten(),
			keras.layers.Dense(1028, activation=tf.nn.relu),
			keras.layers.Dropout(0.25),
			keras.layers.Dense(256, activation=tf.nn.relu),
			keras.layers.Dense(2, activation=tf.nn.softmax)	
		])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit(training_x, training_y, epochs=10)

# Validate
pred = model.predict(validation_x)
probs_pos = [prob[1] for prob in pred]
probs_pos = np.asarray(probs_pos)
roc_auc = roc_auc_score(validation_y, probs_pos)
print(roc_auc)

# Save submission
predictions = model.predict(x_test)
solution = [prob[1] for prob in predictions]
save_solution(my_solution_file, solution)

