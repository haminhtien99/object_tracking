import cv2
import matplotlib.pyplot as plt
import os

import sys
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from modules.take_roi import take_roi

# import os


folder = 'C:/Users/ABC/Desktop/VisDrone2019-VID-val/sequences/uav0000339_00001_v'

list_jpg = os.listdir(folder)
filenames = [f for f in list_jpg]
filenames.sort()
full_paths=[]

for f in filenames:
    
    full_paths.append(folder+'/'+f)
roi = take_roi(full_paths[0])
roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

methods = ['cv2.TM_SQDIFF_NORMED', 'cv2.TM_CCORR_NORMED', 'cv2.TM_CCOEFF_NORMED']
method = methods[2]
for file in full_paths[1:]:
    img = cv2.imread(file)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    R = cv2.matchTemplate(img_gray, roi, eval(method))
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(R)
    if method =='cv2.TM_SQDIFF_NORMED':
        top_left = min_loc
        if min_val < 0.05:
            bottom_right = (top_left[0] + len(roi[0]), top_left[1] + len(roi))
            # print(top_left,min_val,max_val )
            cv2.rectangle(img, top_left, bottom_right, 0,2)
    else: 
        top_left = max_loc
        if max_val >0.55:
            bottom_right = (top_left[0] + len(roi[0]), top_left[1] + len(roi))
            cv2.rectangle(img, top_left, bottom_right, 0,2)
    cv2.imshow("Tracking", img)
    #press Esc to quit
    k = cv2.waitKey(1) & 0xff
    if k == 27 : break

cv2.destroyAllWindows()