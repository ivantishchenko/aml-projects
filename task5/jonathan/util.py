import biosppy
import numpy as np

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