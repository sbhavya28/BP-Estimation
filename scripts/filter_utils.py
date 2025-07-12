# scripts/filter_utils.py

import numpy as np
from scipy.signal import find_peaks

FS = 125
MIN_LENGTH = 800
STD_THRESHOLD = 0.01
MIN_PEAKS = 5
PEAK_DISTANCE_SEC = 0.5

def is_short_signal(signal, fs=125, window_sec=8, overlap_ratio=0.5):
    """
    Dynamically calculate minimum length required for at least one window.
    """
    window_size = int(fs * window_sec)
    step = int(window_size * (1 - overlap_ratio))
    if len(signal) < window_size:
        return True
    return False

def is_flat_signal(signal, std_thresh=STD_THRESHOLD):
    return np.std(signal) < std_thresh

def has_insufficient_peaks(signal, fs=FS, min_peaks=MIN_PEAKS):
    distance_samples = int(PEAK_DISTANCE_SEC * fs)
    peaks, _ = find_peaks(signal, distance=distance_samples)
    return len(peaks) < min_peaks

def is_ppg_signal_acceptable(ppg_signal):
    """
    Returns True if the signal passes all quality checks.
    """
    if is_short_signal(ppg_signal):
        return False, "Too short"
    if is_flat_signal(ppg_signal):
        return False, "Too flat (low std)"
    if has_insufficient_peaks(ppg_signal):
        return False, "Insufficient peaks"
    return True, "OK"
