
import numpy as np
import cv2
def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rows > 1:
        for x in range(0, rows):
            for y in range(0,cols):
                if imgArray[x][y].shape[:2] != imgArray[0][0].shape[:2]:
                    imgArray[x][y]= cv2.resize(imgArray[x][y], (width, height))
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        # ver = [imageBlank] * cols
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for y in range(0, cols):
            if imgArray[0][y].shape[:2] != imgArray[0][0].shape[:2]:
                    imgArray[0][y]= cv2.resize(imgArray[0][y], (width, height))
            if len(imgArray[0][y].shape) == 2: imgArray[0][y] = cv2.cvtColor(imgArray[0][y], cv2.COLOR_GRAY2BGR)
        ver = np.hstack(imgArray[0])
    
    ver = cv2.resize(ver, (int(width *cols *scale), int(height* rows*scale)))
    
    return ver