from scipy.signal import butter, filtfilt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pywt

FS = 125

def bandpass_filter(signal, lowcut=0.5, highcut=8.0, order=3):
    nyquist = 0.5 * FS
    low, high = lowcut / nyquist, highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def wavelet_denoise(signal, wavelet='db6', level=3):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(signal)))
    denoised_coeffs = [coeffs[0]] + [
        pywt.threshold(c, value=uthresh, mode='soft') for c in coeffs[1:]
    ]
    return pywt.waverec(denoised_coeffs, wavelet)

def remove_baseline(signal, cutoff=0.3):
    b, a = butter(2, cutoff / (0.5 * FS), btype='highpass')
    return filtfilt(b, a, signal)

def normalize_signal(signal):
    scaler = MinMaxScaler()
    return scaler.fit_transform(signal.reshape(-1, 1)).flatten()

def preprocess_ppg_signal(ppg_raw):
    ppg_filtered = bandpass_filter(ppg_raw)
    ppg_denoised = wavelet_denoise(ppg_filtered)
    ppg_baseline_removed = remove_baseline(ppg_denoised)
    ppg_normalized = normalize_signal(ppg_baseline_removed)
    return ppg_normalized
