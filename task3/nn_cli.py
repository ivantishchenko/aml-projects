import keras
import keras.layers as layers
import keras.activations as activations
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import transformations

def feed(X, y, training=True, shorten=False, batch_size=128, spectrogram=False):
    seqlens = np.isnan(X[:, :, 0]).argmax(axis=1)
    seqlens[seqlens == 0] = X.shape[1]
    target_len = seqlens.min()
    print('Shortening to %d' % target_len)

    # Loop over epochs
    while True:
        idx = np.arange(X.shape[0])

        if training:
            np.random.shuffle(idx)

        # Loop over batches
        for k in range(0, X.shape[0], batch_size):
            X_batch, y_batch = [], []

            # Loop over examples
            for j in range(k, min(X.shape[0], k + batch_size)):
                i = idx[j]
                example = np.copy(X[i])
                seqlen = seqlens[i]

                if training:
                    while True:
                        # Randomly rescale
                        example = transformations.random_resample(example[np.newaxis, :seqlen, 0]).T
                        seqlen = np.isnan(example).argmax()
                        if seqlen == 0: seqlen = example.shape[0]
                        if seqlen >= target_len: break
                        example = np.copy(X[i])
                        seqlen = seqlens[i]

                    # Randomly set runs to zero
                    for _ in range(int(seqlens[i] / X.shape[1] * 50)):
                        blen = np.random.randint(50, 250)
                        pos = np.random.randint(seqlens[i] - blen)
                        example[pos:pos + blen] = 0

                if training and not shorten:
                    # Repeat sequence to fill padding
                    example = np.tile(example[:seqlen], [int(np.ceil(X.shape[1] / seqlen)), 1])
                    example = example[:X.shape[1]]
                elif training and shorten:
                    # Choose random window to shorten sequence
                    offset = np.random.randint(seqlen - target_len + 1)
                    example = example[offset:offset + target_len]

                if spectrogram:
                    # Convert into spectrogram
                    example = transformations.spectrogram(example[np.newaxis, :seqlen, 0], 64)[0, :, :, np.newaxis]

                X_batch.append(example)

                if y is not None:
                    y_batch.append(y[i])

            if y is not None:
                yield np.array(X_batch), np.array(y_batch)
            else:
                yield np.array(X_batch)

# Load the data
X_test = np.load('X_test.npy')
X_train = np.load('X_train.npy')
y_train = np.load('Y_train.npy')

# Add channel dimension
X_train = X_train[:, :, np.newaxis]
X_test = X_test[:, :, np.newaxis]

# Convert to one-hot
y_train = keras.utils.to_categorical(y_train[:, 1])

# Create validations set
X_train, X_val, Y_train, Y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

model = keras.models.Sequential()
# model.add(layers.Dropout(.5, input_shape=(None, 33, 1)))

for _ in range(3):
    model.add(layers.Conv2D(64, 3, padding='same', input_shape=(None, 33, 1)))
    model.add(layers.Conv2D(64, 3, padding='same'))
    model.add(layers.Conv2D(64, 3, padding='same'))
    model.add(layers.Activation('relu'))
    model.add(keras.layers.normalization.BatchNormalization())

    # Maxpool over time only
    model.add(layers.Reshape((-1, 33 * 64)))
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Reshape((-1, 33, 64)))

# Global maxpool over time only
model.add(layers.Reshape((-1, 33 * 64)))
model.add(layers.GlobalMaxPooling1D())

model.add(layers.Dense(2048, activation='relu'))
model.add(keras.layers.normalization.BatchNormalization())
model.add(layers.Dense(2048, activation='relu'))
model.add(layers.Dense(4, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit_generator(feed(X_train, Y_train, shorten=True, spectrogram=True),
                    steps_per_epoch=np.ceil(X_train.shape[0]/128),
                    epochs=4000,
                    validation_data=feed(X_val, Y_val, shorten=True, batch_size=1, spectrogram=True, training=False),
                    validation_steps=X_val.shape[0],
                    callbacks=[
                        keras.callbacks.EarlyStopping(monitor='val_acc', patience=50, restore_best_weights=True)
                    ])

model.evaluate_generator(feed(X_val, Y_val, shorten=True, batch_size=1, spectrogram=True, training=False),
                         steps=X_val.shape[0])

# Predict
y_test = model.predict_generator(feed(X_test, None, training=False, spectrogram=True, batch_size=1), steps=X_test.shape[0])
assert y_test.shape[0] == X_test.shape[0]
np.savetxt("Y_test.csv", np.stack(( np.arange(X_test.shape[0]), y_test.argmax(axis=1) ), axis=1),
           delimiter=",", header="id,y", fmt='%d', comments='')