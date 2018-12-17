import biosppy
import numpy as np
import scipy
import mne
import pywt
import math
import sys
from tqdm import tqdm

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
    bp = np.sum(psd[:, idx_band], axis=1)
    return bp

def bandstd(psd, freqs, band):
    low = band[0]
    high = band[1]
    idx_band = np.logical_and(freqs >= low, freqs < high)
    return np.std(psd[:, idx_band], axis=1)

def bandsquaredmean(psd, freqs, band):
    idx_band = np.logical_and(freqs >= band[0], freqs < band[1])
    return (psd[:, idx_band] * psd[:, idx_band]).mean(axis=1)

def get_band_features(eeg):
    bands = [[0.39, 3.13], [3.13, 8.46], [8.46, 10.93], [10.93, 15.63], [15.63, 21.88], [21.88, 37.50]]
    features = []
    psd, freqs = mne.time_frequency.psd_array_multitaper(eeg, 128., adaptive=True, normalization='full', n_jobs=16)

    for j in tqdm(range(len(bands))):
        power = bandpower(psd, freqs, bands[j])
        stdev = bandstd(psd, freqs, bands[j])
        squaredmean = bandsquaredmean(psd, freqs, bands[j])
        relativepower = bandpower(np.absolute(psd), freqs, bands[j]) / np.sum(np.absolute(psd))
        features.append(power)
        features.append(stdev)
        features.append(squaredmean)

    alphapower = bandpower(psd, freqs, bands[2])
    deltapower = bandpower(psd, freqs, bands[0])
    thetapower = bandpower(psd, freqs, bands[1])

    features.append(alphapower / (deltapower + thetapower))
    features.append(deltapower / (alphapower + thetapower))
    features.append(thetapower / (deltapower + alphapower))
        
    features.append(np.max(np.absolute(eeg), axis=1))
    features.append(np.sum(np.absolute(eeg), axis=1))

    return np.asarray(features).T

def get_emg_features(emg):
    feature_list = []
    for i in tqdm(range(emg.shape[0])):
        features = []
        features.append(emg[i].std())
        features.append(np.absolute(emg[i]).mean())
        features.append(np.max(np.absolute(emg[i])) - np.min(np.absolute(emg[i])))
        feature_list.append(np.asarray(features))
    return np.asarray(feature_list)

def get_wavelet_coeffs(eeg):
    coeffs = []
    for i in tqdm(range(eeg.shape[0])):
        coeff = pywt.wavedec(eeg[i], 'db4', level=5)
        coeffs.append(coeff)
    return coeffs

def get_wavelet_features(wavelets):
    feature_list = []
    for i in tqdm(range(len(wavelets))):
        features = []
        coeff = wavelets[i]
        for j in range(len(coeff)):
            c = np.asarray(coeff[j])
            absc = np.absolute(c)
            features.append(np.sum(absc * absc))
            features.append(c.std())
            features.append(absc.mean())
            features.append(np.max(c))

        feature_list.append(np.asarray(features))
    return np.asarray(feature_list)

def reslice(data, labels):
    # Extract runs
    run_data, run_labels = [], []
    current_label, current_run = labels[0], [data[0]]

    for data_slice, label in zip(data[1:], labels[1:]):
        if label == current_label:
            current_run.append(data_slice)
        else: 
            run_data.append(np.concatenate(current_run))
            run_labels.append(current_label)
            current_label, current_run = label, [data_slice]
    run_data.append(np.concatenate(current_run))
    run_labels.append(current_label)

    # Slice runs
    slice_length = data.shape[1]
    slices, slice_labels = [], []
    for data_slice, label in zip(run_data, run_labels):
        pos = 0
    
        while pos + slice_length <= len(data_slice):
            slices.append(data_slice[pos:pos + slice_length])
            slice_labels.append(label)
            pos += np.random.randint(slice_length // 4, slice_length // 2)
        
        if pos != len(data_slice):
            slices.append(data_slice[-slice_length:])
            slice_labels.append(label)
            
    return np.array(slices), np.array(slice_labels)
