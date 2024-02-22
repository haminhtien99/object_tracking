from matplotlib import pyplot as plt
import argparse
import cv2
import numpy as np

import warnings
warnings.filterwarnings("ignore")

path = 'C:/Users/ABC/Desktop/object_tracking/sample_images/0000001.jpg'
img = cv2.imread(path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Normalize the grayscale image
# normalized = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
equ = cv2.equalizeHist(img)
# Apply a threshold to the normalized image
# _, threshold = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
_, threshold = cv2.threshold(equ, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

cv2.imshow('img with threshold',cv2.cvtColor(threshold, cv2.COLOR_GRAY2BGR))
hist = cv2.calcHist([threshold], [0], None, [256], [0, 256])
plt.figure()
plt.plot(hist)
plt.title("hist of threshold img")
plt.xlim([-1,300])
plt.show()

contours, hierarchy = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
largest_contour = max(contours, key=cv2.contourArea)
largest_area = cv2.contourArea(largest_contour)
if largest_area > 0.9 *len(img) * len(img[0]): 
    contours, hierarchy = cv2.findContours(~threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    largest_area = cv2.contourArea(largest_contour)
    if largest_area > 0.9 *len(img) * len(img[0]): 
        print('false')
    else: print('white ground')
else: print('black ground')

roi_with_contour = img.copy()
for contour in contours:
    area = cv2.contourArea(contour)
    if area > 100:
        x_min = min(contour[:, 0, 0])
        x_max = max(contour[:, 0, 0])
        y_min = min(contour[:, 0, 1])
        y_max = max(contour[:, 0, 1])
        cv2.rectangle(roi_with_contour,(x_min,y_min),(x_max,y_max),(0,255,0),1)
        
cv2.imshow('objects in img',cv2.cvtColor(roi_with_contour , cv2.COLOR_GRAY2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()    
