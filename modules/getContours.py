import os
import sys
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from modules.stackImages import stackImages as stk
import cv2
import numpy as np

def getContour(img, minArea = 50):
    unique_values = np.unique(img)
    if len(unique_values) == 2:   
        imgCanny = img
        # imgGray = img
    else:
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgBlur = cv2.GaussianBlur(imgGray, (7,7), 1)
        imgCanny = cv2.Canny(imgBlur, 50, 50)

    imgCopy = img.copy()
    contours, _ = cv2.findContours(imgCanny, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    boudingboxs = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > minArea:
            cv2.drawContours(imgCopy, contour, -1, (0,0,0), 3)
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02* peri, True)
            x, y, w, h = cv2.boundingRect(approx)
            boudingboxs.append([x, y, w, h])
            cv2.rectangle(imgCopy, (x, y), (x + w, y + h), (0, 0, 255), 3)
        
    return imgCopy, boudingboxs

def getMaxContour(img, minArea = 50):
    unique_values = np.unique(img)
    if len(unique_values) == 2:   
        imgCanny = img
        # imgGray = img
    else:
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgBlur = cv2.GaussianBlur(imgGray, (7,7), 1)
        imgCanny = cv2.Canny(imgBlur, 50, 50)

    # imgCopy = img.copy()
    contours, _ = cv2.findContours(imgCanny, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    MaxArea = 0
    biggest =np.array([])
    con = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > minArea:
            # cv2.drawContours(imgCopy, contour, -1, (0,0,0), 3)
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02* peri, True)
            if area > MaxArea and len(approx) == 4:
                MaxArea = area
                biggest = approx
                con = contour
    return biggest, con            
if __name__ == "__main__":
    path = 'C:/Users/ABC/Desktop/object_tracking/sample_images/original-1384640-2.jpg'
    img = cv2.imread(path)
    
    result, _ = getContour(img, 50)
    cv2.imshow("result", stk(0.5, [[img, result]]))
    cv2.waitKey(0)