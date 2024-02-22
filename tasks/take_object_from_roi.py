from matplotlib import pyplot as plt
import argparse
import cv2
import numpy as np

import warnings
warnings.filterwarnings("ignore")

def take_roi(path_to_img):
    img = cv2.imread(path_to_img)
    imgResize = cv2.resize(img, (1000,1000))
    print(img.shape[0], img.shape[1])
    bbox = cv2.selectROI('Select ROI',imgResize)
    
    # x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
    x = int(bbox[0] * img.shape[1]/1000)
    y = int(bbox[1] * img.shape[0]/1000)
    w= int(bbox[2] * img.shape[1]/1000)
    h = int(bbox[3] * img.shape[0]/1000)
    
    roi = img[y: y+h,x:x+w]
    # roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    cv2.destroyAllWindows()
    return roi


path = 'C:/Users/ABC/Desktop/object_tracking/sample_images/0000001.jpg'
img = cv2.imread(path)
roi = take_roi(path)
roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

# Normalize the grayscale image
normalized = cv2.normalize(roi, None, 100, 255, cv2.NORM_MINMAX)
# Apply a threshold to the normalized image
_, threshold = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imshow('roi with threshold',cv2.cvtColor(threshold, cv2.COLOR_GRAY2BGR))
hist = cv2.calcHist([threshold], [0], None, [256], [0, 256])
plt.figure()
plt.plot(hist)
plt.title("hist of threshold img")
plt.xlim([-1,300])
plt.show()

contours, hierarchy = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
largest_contour = max(contours, key=cv2.contourArea)
largest_area = cv2.contourArea(largest_contour)
if largest_area > 0.9 *len(roi) * len(roi[0]): 
    contours, hierarchy = cv2.findContours(~threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    largest_area = cv2.contourArea(largest_contour)
    if largest_area > 0.9 *len(roi) * len(roi[0]): 
        print('false')
    else: print('white ground')
else: print('black ground')

roi_with_contour = roi.copy()
for contour in contours:
    area = cv2.contourArea(contour)
    if area > 100:
        x_min = min(contour[:, 0, 0])
        x_max = max(contour[:, 0, 0])
        y_min = min(contour[:, 0, 1])
        y_max = max(contour[:, 0, 1])
        cv2.rectangle(roi_with_contour,(x_min,y_min),(x_max,y_max),(0,255,0),1)
        
cv2.imshow('objects in roi',cv2.cvtColor(roi_with_contour , cv2.COLOR_GRAY2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()    
