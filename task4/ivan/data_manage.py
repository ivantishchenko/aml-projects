import numpy as np
import math

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

# Only on the horizontal axis
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