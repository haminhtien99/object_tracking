import numpy as np
import cv2
import os


folder = 'C:/Users/ABC/Desktop/VisDrone2019-VID-val/sequences/uav0000268_05773_v'

list_jpg = os.listdir(folder)
filenames = [f for f in list_jpg]
filenames.sort()
full_paths=[]

for f in filenames:
    
    full_paths.append(folder+'/'+f)
# cap = cv2.VideoCapture('cars.mp4')
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
for path in full_paths:
    # ret, frame = cap.read()
    frame =cv2.imread(path)

    fgmask = fgbg.apply(frame)
    cv2.imshow('frame', fgmask)
    k = cv2.waitKey(30) & 0xff
    if k == 27: break
# cap.release()
cv2.destroyAllWindows()