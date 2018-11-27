import numpy as np

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
