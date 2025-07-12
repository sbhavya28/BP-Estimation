# scripts/scalogram_utils.py

import numpy as np
import pywt
import cv2

IMG_SIZE = (128, 128)
WAVELET = 'morl'
SCALES = np.arange(1, 65)

def generate_scalogram_image(ppg_segment, img_size=IMG_SIZE, wavelet=WAVELET):
    """
    Returns scalogram image (numpy array) from PPG segment.
    """
    coeffs, _ = pywt.cwt(ppg_segment, SCALES, wavelet)
    scalogram = np.abs(coeffs)

    # Normalize 0–255
    scalogram -= scalogram.min()
    if scalogram.max() != 0:
        scalogram /= scalogram.max()
    scalogram_img = (scalogram * 255).astype(np.uint8)

    # Resize
    scalogram_img = cv2.resize(scalogram_img, img_size)
    return scalogram_img
