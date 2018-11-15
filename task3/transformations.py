import numpy as np
import scipy as sc
from scipy import signal


def spectrogram(data, nperseg=32, noverlap=16):
    log_spectrogram = True
    fs = 300
    _, _, Sxx = signal.spectrogram(data, fs=fs, nperseg=nperseg, noverlap=noverlap)
    Sxx = np.transpose(Sxx, [0, 2, 1])
    if log_spectrogram:
        Sxx = abs(Sxx)
        mask = Sxx > 0
        Sxx[mask] = np.log(Sxx[mask])
    return Sxx


def random_resample(signals, upscale_factor=1):
    [n_signals, length] = signals.shape
    new_length = np.random.randint(
        low=int(length * 80 / 120),
        high=int(length * 80 / 60),
        size=[n_signals, upscale_factor]
    )
    signals = [np.array(s) for s in signals.tolist()]
    new_length = [np.array(nl) for nl in new_length.tolist()]
    sigs = [stretch_squeeze(s, l) for s, nl in zip(signals, new_length) for l in nl]
    sigs = [fit_tolength(s, length) for s in sigs]
    sigs = np.array(sigs)
    return sigs


def stretch_squeeze(source, length):
    target = np.zeros([1, length])
    interpol_obj = sc.interpolate.interp1d(np.arange(source.size), source)
    grid = np.linspace(0, source.size - 1, target.size)
    result = interpol_obj(grid)
    return result


def fit_tolength(source, length):
    target = np.ones([length]) * np.nan
    w_l = min(source.size, target.size)
    target[0:w_l] = source[0:w_l]
    return target