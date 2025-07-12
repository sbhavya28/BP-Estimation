# scripts/segment_utils.py

import numpy as np

FS = 125
WINDOW_SEC = 8
OVERLAP_RATIO = 0.5

def segment_ppg_abp(ppg, abp, fs=FS, window_sec=WINDOW_SEC, overlap_ratio=OVERLAP_RATIO):
    window_size = int(fs * window_sec)
    step = int(window_size * (1 - overlap_ratio))
    X_ppg, Y_labels = [], []

    for start in range(0, len(ppg) - window_size + 1, step):
        end = start + window_size
        seg_ppg = ppg[start:end]
        seg_abp = abp[start:end]

        if np.std(seg_ppg) < 0.01:
            continue

        sbp = np.max(seg_abp)
        dbp = np.min(seg_abp)
        X_ppg.append(seg_ppg)
        Y_labels.append([sbp, dbp])

    return np.array(X_ppg), np.array(Y_labels)
