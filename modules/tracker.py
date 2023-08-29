from matplotlib import pyplot as plt
import argparse
import cv2
import numpy as np
import os

import warnings
warnings.filterwarnings("ignore")

from modules.take_roi import take_roi
def tracker(roi):
    
    # Normalize the grayscale image
    normalized = cv2.normalize(roi, None, 100, 255, cv2.NORM_MINMAX)
    # Apply a threshold to the normalized image
    _, threshold = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    contours, hierarchy = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    largest_area = cv2.contourArea(largest_contour)
    if largest_area > 0.9 *len(roi) * len(roi[0]): 
        contours, hierarchy = cv2.findContours(~threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        largest_area = cv2.contourArea(largest_contour)
        if largest_area > 0.9 *len(roi) * len(roi[0]): 
            print('false')
        # else: print('white ground')
    # else: print('black ground')
    
    roi_with_contour = roi.copy()
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:
            x_min = min(contour[:, 0, 0])
            x_max = max(contour[:, 0, 0])
            y_min = min(contour[:, 0, 1])
            y_max = max(contour[:, 0, 1])
            cv2.rectangle(roi_with_contour,(x_min,y_min),(x_max,y_max),(0,255,0),1)
    plt.figure()
    plt.imshow(cv2.cvtColor(roi_with_contour , cv2.COLOR_GRAY2BGR))    