import os
import sys
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
# from modules.stackImages import stackImages as stk
from modules.getContours import getMaxContour
from modules.stackImages import stackImages as stk
import cv2
import numpy as np

def imgPreprocessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (5, 5), 1)
    img = cv2.Canny(img, 200, 200)
    kernel = np.ones((5, 5))
    img = cv2.dilate(img, kernel, iterations=2)
    img = cv2.erode(img, kernel, iterations=1)
    # img = cv2.bitwise_not(img)
    # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img
def warpPerspective(img):
    biggest_contour, _ = getMaxContour(img)
    pts1 = np.float32(biggest_contour)
    pts2 = np.float32([[0, 0], [300, 0] ,[0, 400], [300, 400]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgOutput = cv2.warpPerspective(img, matrix, (300, 400))
    return imgOutput

path = 'C:/Users/ABC/Desktop/object_tracking/sample_images/A4-Paper-PSD-MockUp-6001.jpg'
img = cv2.imread(path)
# biggest_contour, con = getMaxContour(img)
# print(biggest_contour)
# cv2.drawContours(img, con, -1, (0,0,0), 3)
# cv2.drawContours(img, biggest_contour, -1, (0,0,255), 3)
img = imgPreprocessing(img)
# paperScan = warpPerspective(img)
# result = stk)
cv2.imshow('image', img)
cv2.waitKey(0)



