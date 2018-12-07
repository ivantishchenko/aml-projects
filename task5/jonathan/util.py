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

def bandpower(psd, freqs, band):
    low = band[0]
    high = band[1]
    idx_band = np.logical_and(freqs >= low, freqs < high)
    bp = np.sum(psd[idx_band])
    return bp

def bandstd(psd, freqs, band):
    low = band[0]
    high = band[1]
    idx_band = np.logical_and(freqs >= low, freqs < high)
    return psd[idx_band].std()

def bandsquaredmean(psd, freqs, band):
    idx_band = np.logical_and(freqs >= band[0], freqs < band[1])
    return (psd[idx_band] * psd[idx_band]).mean()

def get_band_features(eeg):
    feature_list = []
    bands = [[0.39, 3.13], [3.13, 8.46], [8.46, 10.93], [10.93, 15.63], [15.63, 21.88], [21.88, 37.50]]
    for i in range(eeg.shape[0]):
        print("Computing features for : " + str(i))
        features = []
        signal = eeg[i]
        psd, freqs = mne.time_frequency.psd_array_multitaper(eeg[i], 128., adaptive=True, normalization='full')
        
        for j in range(len(bands)):
            power = bandpower(psd, freqs, bands[j])
            stdev = bandstd(psd, freqs, bands[j])
            squaredmean = bandsquaredmean(psd, freqs, bands[j]) 
            features.append(power)
            features.append(stdev)
            features.append(squaredmean)

        alphapower = bandpower(psd, freqs, bands[2])
        deltapower = bandpower(psd, freqs, bands[0])
        thetapower = bandpower(psd, freqs, bands[1])

        features.append(alphapower / (deltapower + thetapower))
        features.append(deltapower / (alphapower + thetapower))
        features.append(thetapower / (deltapower + alphapower))

        features.append(np.max(np.absolute(eeg[i])))
        features.append(np.sum(np.absolute(eeg[i])))

        feature_list.append(np.asarray(features))
    return np.asarray(feature_list)

