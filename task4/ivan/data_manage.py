import numpy as np
import math
import matplotlib.pyplot as plt
import scipy
from skimage.transform import resize


# Extract blocks of training data from videos [4, 100, 100] non-overlapping
# and return list of these and a list of all labels 
def sliding_training_data(videos, y, blocksize=4):
    blocks = []
    labels = []
    for i in range(videos.shape[0]):
        video = videos[i]
        nframes = video.shape[0]
        for j in range(nframes // blocksize - 1):
            startIndex = blocksize * j
            endIndex = blocksize * (j + 1)
            block = video[startIndex:endIndex, :, :]
            block = block.reshape((1, blocksize, 100, 100))
            blocks.append(block)
            labels.append(y[i])
    return np.asarray(blocks), np.asarray(labels)


# Do rotations
def shear(videos, labels):
    vids = videos
    labs = labels
    hshear = videos
    for i in range(videos.shape[0]):
        # # PLOT VID
        # demo = hshear[i][5]
        # plt.imshow(demo, cmap='gray')
        # plt.show()
        # # print(hshear[i])
        # # PLOT VID

        shear_len = np.random.uniform(-0.12, 0.12, 1)[0]

        shear_matrix = np.array([[1, 0, 0, 0],
                                 [0, 1, shear_len, 0],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, 1]])
        hshear[i] = scipy.ndimage.affine_transform(hshear[i], shear_matrix)

        # # PLOT VID
        # # print(hshear[i])
        # demo = hshear[i][5]
        # plt.imshow(demo, cmap='gray')
        # plt.show()
        # # PLOT VID

    vids = np.concatenate((vids, hshear), axis=0)
    labs = np.concatenate((labs, labels), axis=0)
    return vids, labs


# Do rotations
def zoom(videos, labels):
    vids = videos
    labs = labels
    hscale = videos
    for i in range(videos.shape[0]):
        # # PLOT VID
        # demo = hscale[i][5]
        # plt.imshow(demo, cmap='gray')
        # plt.show()
        # # print(hscale[i])
        # # PLOT VID

        factor = np.random.uniform(-0.08, 0.08, 1)[0]
        hscale[i] = scipy.ndimage.rotate(hscale[i], factor)

        # # PLOT VID
        # # print(hscale[i])
        # demo = hscale[i][5]
        # plt.imshow(demo, cmap='gray')
        # plt.show()
        # # PLOT VID

    vids = np.concatenate((vids, hscale), axis=0)
    labs = np.concatenate((labs, labels), axis=0)
    return vids, labs


# Shift horizontal/vertical
def shift(videos, labels, direction):
    vids = videos
    labs = labels
    hshift = videos
    for i in range(videos.shape[0]):
        # # PLOT VID
        # demo = hshift[i][5]
        # plt.imshow(demo, cmap='gray')
        # plt.show()
        # # print(hshift[i])
        # # PLOT VID

        shift_len = np.random.randint(-10, 11, size=1)

        if direction == 'vertical':
            hshift[i] = scipy.ndimage.shift(hshift[i], (0, shift_len, 0))
        if direction == 'horizontal':
            hshift[i] = scipy.ndimage.shift(hshift[i], (0, 0, shift_len))

        # # PLOT VID
        # # print(hshift[i])
        # demo = hshift[i][5]
        # plt.imshow(demo, cmap='gray')
        # plt.show()
        # # PLOT VID

    vids = np.concatenate((vids, hshift), axis=0)
    labs = np.concatenate((labs, labels), axis=0)
    return vids, labs


# Do rotations
def rotate(videos, labels):
    vids = videos
    labs = labels
    hrotate = videos
    for i in range(videos.shape[0]):
        # # PLOT VID
        # demo = hrotate[i][5]
        # plt.imshow(demo, cmap='gray')
        # plt.show()
        # # print(hrotate[i])
        # # PLOT VID

        angle = np.random.randint(-10, 11, size=1)
        hrotate[i] = scipy.ndimage.rotate(hrotate[i], angle, axes=(1, 2), reshape=False)

        # # PLOT VID
        # # print(hrotate[i])
        # demo = hrotate[i][5]
        # plt.imshow(demo, cmap='gray')
        # plt.show()
        # # PLOT VID

    vids = np.concatenate((vids, hrotate), axis=0)
    labs = np.concatenate((labs, labels), axis=0)
    return vids, labs


def flip(videos, labels, horizontal=True, frames=False):
    vids = videos
    labs = labels
    if horizontal:
        hflipped = videos
        for i in range(videos.shape[0]):
            hflipped[i] = np.flip(hflipped[i], axis=2)
        vids = np.concatenate((vids, hflipped), axis=0)
        labs = np.concatenate((labs, labels), axis=0)
    if frames:
        fflipped = videos
        for i in range(videos.shape[0]):
            fflipped[i] = np.flip(fflipped[i], axis=0)
        vids = np.concatenate((vids, fflipped), axis=0)
        labs = np.concatenate((labs, labels), axis=0)
    return vids, labs


def normalize_data(X):
    X_normed = X
    for i in range(X.shape[0]):
        X_normed[i] = (X[i] - 127.5) / 255.0
    return X_normed


def extend_videos(X, extension=216):
    X_extended = []

    for i in range(X.shape[0]):
        tiles = (extension - X[i].shape[0]) / X[i].shape[0]
        tiles = math.ceil(tiles) + 1
        ext = np.tile(X[i], (tiles, 1, 1))
        ext = ext[0:extension, :, :]
        X_extended.append(ext)
    X_extended = np.asanyarray(X_extended)
    return X_extended


def get_frames(X, y):
    frames = []
    labels = []
    for i in range(X.shape[0]):
        for j in range(X[i].shape[0]):
            frame = X[i][j, :, :]
            frame = frame.reshape((100, 100, 1))
            frames.append(frame)
            labels.append(y[i])
    return np.asarray(frames), np.asarray(labels)
