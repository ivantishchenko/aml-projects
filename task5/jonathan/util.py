import biosppy
import numpy as np
import scipy
import mne

def create_eeg(eeg):
    signals = []
    for i in range(eeg.shape[0]):
        sg = np.transpose(np.asarray([eeg[i]]))
        signal = biosppy.signals.eeg.eeg(signal=sg, sampling_rate=128.0, show=False)
        signals.append(signal)
    return signals

def get_eeg_features(signals):
    feature_list = []
    for i in range(len(signals)):
        features = []
        theta = np.asanyarray(signals[i]["theta"])
        alpha_low = np.asanyarray(signals[i]["alpha_low"])
        alpha_high = np.asanyarray(signals[i]["alpha_high"])
        beta = np.asanyarray(signals[i]["beta"])
        gamma = np.asanyarray(signals[i]["gamma"])

        features.append(theta.mean())
        features.append(theta.std())
        features.append(alpha_low.mean())
        features.append(alpha_low.std())
        features.append(alpha_high.mean())
        features.append(alpha_high.std())
        features.append(beta.mean())
        features.append(beta.std())
        features.append(gamma.mean())
        features.append(gamma.std())

        features = np.asarray(features)
        feature_list.append(features)

    return np.asarray(feature_list)

def get_band_features(eeg):
    feature_list = []
    bands = [[0.5, 4], [4, 8], [8, 12], [12, 16], [16, 32]]
    for i in range(eeg.shape[0]):
        print("Computing features for : " + str(i))
        features = []
        signal = eeg[i]
        psd, freqs = mne.time_frequency.psd_array_multitaper(signal, 128., adaptive=True, normalization='full')
        for j in range(len(bands)):
            low = bands[j][0]
            high = bands[j][1]
            idx = np.logical_and(freqs >= low, freqs < high)
            bp = np.sum(psd[idx])
            rbp = bp / np.sum(psd)
            features.append(bp)
            features.append(rbp)
        feature_list.append(np.asarray(features))
    return np.asarray(feature_list)

