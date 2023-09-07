import os
import sys
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from modules.colorDetection import colorDetection
from modules.getContours import getContour
import cv2
import numpy as np

folder = 'C:/Users/ABC/Desktop/VisDrone2019-VID-val/sequences/uav0000305_00000_v'

list_jpg = os.listdir(folder)
filenames = [f for f in list_jpg]
filenames.sort()
full_paths=[]

for f in filenames:
    
    full_paths.append(folder+'/'+f)
# lowers, upper = colorDetection(full_paths[0])


lower = np.array([42, 124, 156])
upper = np.array([51, 255, 255])
colors = {"Green":[lower, upper]}


for path in full_paths:
    img = cv2.imread(path)
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(imgHSV, lower, upper)
    _, boudingboxs = getContour(mask)
    for box in boudingboxs:
        cv2.rectangle(img, (box[0], box[1]), (box[0]+ box[2], box[1] + box[3]), (0, 0, 255), 3)
    cv2.imshow('img', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

