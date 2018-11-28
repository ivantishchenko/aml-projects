import os
import numpy as np
import math
import keras
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense, Input, InputLayer
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Nadam
from keras.preprocessing.sequence import pad_sequences
from utils import save_solution, get_videos_from_folder, get_target_from_csv
from sklearn.model_selection import train_test_split
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten

MAX_SEQ_LEN = 210
VIDEO_DIM = 100
CHANNEL_NUM = 1

dir_path = os.path.dirname(os.path.realpath(__file__))
train_folder = os.path.join(dir_path,"../train/")
test_folder = os.path.join(dir_path,"../test/")
train_target = os.path.join(dir_path,'../train_target.csv')
my_solution_file = os.path.join(dir_path,'../solution.csv')

# x_train = get_videos_from_folder(train_folder)
# y_train = get_target_from_csv(train_target)
# x_test = get_videos_from_folder(test_folder)
# np.save('../X_train', x_train)
# np.save('../Y_train', y_train)
# np.save('../X_test', x_test)

# load data
x_train = np.load('../X_train.npy')
y_train = np.load('../Y_train.npy')
x_test = np.load('../X_test.npy')

# avg_seq_len = 0
# for seq in x_train:
#     avg_seq_len += seq.shape[0]
#
# avg_seq_len /= x_train.shape[0]
#
# avg_seq_len = 0
# for seq in x_test:
#     avg_seq_len += seq.shape[0]
#
# avg_seq_len /= x_test.shape[0]

X_train = pad_sequences(x_train, maxlen=MAX_SEQ_LEN, padding='post')
X_test = pad_sequences(x_test, maxlen=MAX_SEQ_LEN, padding='post')

X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# CNN PART
video = Input(shape=(MAX_SEQ_LEN, CHANNEL_NUM, VIDEO_DIM, VIDEO_DIM))
frame = Input(shape=(CHANNEL_NUM, VIDEO_DIM, VIDEO_DIM))

x = Conv2D(32, (5, 5), activation='relu', data_format='channels_first')(frame)
# x = MaxPooling2D((2, 2), data_format='channels_first')(x)

# x = Conv2D(64, (3, 3), activation='relu', data_format='channels_first')(x)
# x = MaxPooling2D((2, 2), data_format='channels_first')(x)
cnn_out = GlobalAveragePooling2D(data_format='channels_first')(x)

cnn = Model([frame], [cnn_out])

# LSTM PART
encoded_frames = TimeDistributed(cnn)(video)
encoded_sequence = LSTM(64)(encoded_frames)
hidden_layer = Dense(units=102, activation="relu")(encoded_sequence)
outputs = Dense(units=2, activation="softmax")(hidden_layer)

model = Model([video], [outputs])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit(X_train, y_train, epochs=10, batch_size=4)