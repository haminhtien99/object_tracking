
# Bị ảnh hưởng nhiều bởi ánh sáng
# Kích thước vật thể thay đổi thì không nhận diện được
# Nhầm lẫn khi bắt bám vật thể cho dù vật thể đứng yên
import numpy as np
import cv2
import os


folder = 'C:/Users/ABC/Desktop/VisDrone2019-VID-val/sequences/uav0000305_00000_v'

list_jpg = os.listdir(folder)
filenames = [f for f in list_jpg]
filenames.sort()
full_paths=[]

for f in filenames:
    
    full_paths.append(folder+'/'+f)
img = cv2.imread(full_paths[0])
bbox = cv2.selectROI(img, False)
x,y,w,h = bbox[0], bbox[1], bbox[2], bbox[3]
cv2.destroyAllWindows()
track_window = x, y, w, h
roi = img[y: y+h,x:x+w]
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
# roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])

cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 10)
for path in full_paths[1:]:
    frame = cv2.imread(path)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0], roi_hist, [0,180], 1)
    # apply meanshift to get the new location
    ret, track_window = cv2.meanShift(dst, track_window, term_crit)
    # Draw it on image
    x,y,w,h = track_window
    img2 = cv2.rectangle(frame, (x,y), (x+w,y+h), 255, 2)
    cv2.imshow('img2', img2)
    k = cv2.waitKey(1) & 0xff
    if k == 27 : break