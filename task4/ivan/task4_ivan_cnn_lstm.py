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
from sklearn.metrics import roc_auc_score

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

# X_train = pad_sequences(x_train, maxlen=MAX_SEQ_LEN, padding='post')
# X_test = pad_sequences(x_test, maxlen=MAX_SEQ_LEN, padding='post')

num_training = math.floor(x_train.shape[0] * 0.8)
np.random.seed(42)
indices = np.random.permutation(x_train.shape[0])
training_idx, validation_idx = indices[:num_training], indices[num_training:]

X_train = x_train[training_idx]
X_val = x_train[validation_idx]
Y_train = y_train[training_idx]
Y_val = y_train[validation_idx]

# CNN PART
video = Input(shape=(None, CHANNEL_NUM, VIDEO_DIM, VIDEO_DIM))
frame = Input(shape=(CHANNEL_NUM, VIDEO_DIM, VIDEO_DIM))

x = Conv2D(32, (5, 5), activation='relu', padding="same", data_format='channels_first')(frame)
x = MaxPooling2D((2, 2), data_format='channels_first')(x)

x = Conv2D(64, (5, 5), activation='relu', padding="same", data_format='channels_first')(x)
x = MaxPooling2D((2, 2), data_format='channels_first')(x)
cnn_out = GlobalAveragePooling2D(data_format='channels_first')(x)

cnn = Model([frame], [cnn_out])

# LSTM PART
encoded_frames = TimeDistributed(cnn)(video)
encoded_sequence = LSTM(256)(encoded_frames)
hidden_layer = Dense(units=1024, activation="relu")(encoded_sequence)
outputs = Dense(units=2, activation="softmax")(hidden_layer)

model = Model([video], [outputs])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

print("Training")
# Training
for i in range(10):
    print('Epoch------------------------------------ ', i + 1)
    epoch_avg = 0
    for j in range(len(X_train)):
        seq = np.array([np.expand_dims(X_train[j], axis=1)])
        label = np.array([Y_train[j]])
        loss = model.train_on_batch(seq, label)
        print("Video {}. Loss = {}".format(j + 1, loss))
        epoch_avg += loss

    epoch_avg /= len(X_train)
    print('Avg Loss = ', epoch_avg)

# Prediction
print("Validation")
predictions_val = []
for j in range(len(X_val)):
    seq = np.array([np.expand_dims(X_val[j], axis=1)])
    prediction = model.predict_on_batch(seq)[0]

    if prediction[0] >= prediction[1]:
        predictions_val.append(0)
    else:
        predictions_val.append(1)

print(predictions_val)

predictions_val = np.asarray(predictions_val)
roc_auc = roc_auc_score(Y_val, predictions_val)
print("Roc Auc = {}".format(roc_auc))