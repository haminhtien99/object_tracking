import cv2
import numpy as np
import os
import sys
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from modules.stackImages import stackImages as stk
def empty(a):
    pass
def colorDetection(path = 'C:/Users/ABC/Desktop/VisDrone2019-VID-val/sequences/uav0000305_00000_v/0000001.jpg'):
    window = "TrackBars"
    cv2.namedWindow(window)
    cv2.resizeWindow(window, 640 ,240)
    cv2.createTrackbar("Hue Min", window, 0, 179,  empty)
    cv2.createTrackbar("Hue Max", window, 0, 179,  empty)
    cv2.createTrackbar("Sat Min", window, 0, 255,  empty)
    cv2.createTrackbar("Sat Max", window, 0, 255,  empty)
    cv2.createTrackbar("Val Min", window, 0, 255,  empty)
    cv2.createTrackbar("Val Max", window, 0, 255,  empty)
    while True:
        img = cv2.imread(path)
        img = cv2.resize(img, (500, 500))
        imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h_min = cv2.getTrackbarPos("Hue Min", window)
        h_max = cv2.getTrackbarPos("Hue Max", window)
        s_min = cv2.getTrackbarPos("Sat Min", window)
        s_max = cv2.getTrackbarPos("Sat Max", window)
        v_min = cv2.getTrackbarPos("Val Min", window)
        v_max = cv2.getTrackbarPos("Val Max", window)
        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])
        mask = cv2.inRange(imgHSV, lower, upper)
        # print(imgHSV.shape)
        result = cv2.bitwise_and(img, img, mask = mask)
        visualization = stk(0.5, [[img, imgHSV], [mask, result]])
        cv2.imshow("what", visualization)
        k = cv2.waitKey(1) & 0xff
        if k == 27 : break
    cv2.destroyAllWindows()
    return lower, upper
if __name__ == '__main__':
    lower, upper = colorDetection()
    print(lower, upper)