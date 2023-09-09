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
    imgGray = cv2.GaussianBlur(img, (5, 5), 1)
    imgCanny = cv2.Canny(imgGray, 100, 200)
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgCanny, kernel, iterations=2)
    imgErode = cv2.erode(imgDial, kernel, iterations=1)
    return imgErode
def reorder(points):
    #shape of pts on warpPerspective : (4,2)
    points = points.reshape((4, 2))
    pointsNew = np.zeros((4,2), np.int32)
    pointsNew[0] = points[np.argmin(points.sum(1))]
    pointsNew[3] = points[np.argmax(points.sum(1))]
    pointsNew[1] = points[np.argmin(np.diff(points, axis=1))]
    pointsNew[2] = points[np.argmax(np.diff(points, axis=1))] 
    return pointsNew   
        
def warpPerspective(img, points):
    
    width, height = 596, 700 # size of a4 paper
    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [width, 0] ,[0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgOutput = cv2.warpPerspective(img, matrix, (width, height))
    return imgOutput

path = 'C:/Users/ABC/Desktop/object_tracking/sample_images/A4-Paper-Mockup-01.jpg'
img = cv2.imread(path)
imgThres = imgPreprocessing(img)
points, con = getMaxContour(imgThres)
# cv2.drawContours(img, points, -1, (0,0,255), 3)
# cv2.drawContours(img, con, -1, (0,0,0), 3)
pointsNew = reorder(points)
paperScan = warpPerspective(img, pointsNew)

cv2.imshow('image', cv2.resize(img, (1000, 1000)))
cv2.imshow('Scan', paperScan)
cv2.waitKey(0)



