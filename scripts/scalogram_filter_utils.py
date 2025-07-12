# scripts/scalogram_filter_utils.py

import cv2
import numpy as np

CONTRAST_THRESHOLD = 50
MEAN_RANGE = (30, 180)
EDGE_THRESHOLD = 70

def has_edges(img, threshold=EDGE_THRESHOLD):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    edge_strength = np.mean(magnitude)
    return edge_strength > threshold

def is_scalogram_acceptable(img, contrast_thresh=CONTRAST_THRESHOLD, mean_range=MEAN_RANGE):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_val = np.mean(gray)
    std_val = np.std(gray)

    if not (mean_range[0] < mean_val < mean_range[1]):
        return False, f"Mean out of range: {mean_val:.2f}"
    if std_val < contrast_thresh:
        return False, f"Low contrast: {std_val:.2f}"
    if not has_edges(img):
        return False, "Insufficient edges"
    return True, "OK"
