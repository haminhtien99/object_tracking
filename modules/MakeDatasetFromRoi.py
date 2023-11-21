import cv2
import numpy as np



# new_images = []
import os
def func():
    path = "C:/Users/ABC/Downloads/00000454.jpg"
    img = cv2.imread(path)
    height = img.shape[0]
    width = img.shape[1]
    x = 138
    y = 109
    w = 101
    h = 98
    roi = img[y: y+h,x:x+w]
    blank= np.zeros_like(img)
    folder = "C:/Users/ABC/Desktop/object_tracking/images_sample/"
    bboxes = []
    for iter in range(1000):
        blank[:, :]  = 255
        if x+w >= width or y + h >= height: break
        x += 1
        y += 1
        for i in range(h):
            for j in range(w):
                blank[y +i, x + j] = roi[i,j]
        imgNum = iter + 287
        frame = iter + 455
        bbox = [x, y, x+w, y+h, 0, 0, imgNum, 0, 0]
        cv2.imwrite(folder +"00000"+ str(frame) + ".jpg", blank)
        bboxes.append(bbox)
    # bboxes = np.array(bboxes)
    x -= 1
    y-= 1
    for i in range(1000):
        blank[:, : ]= 255        
        x-= 1
        if x < 0: break
        for i in range(h):
            for j in range(w):
                blank[y +i, x + j] = roi[i,j]
        imgNum = imgNum + 1
        frame = frame + 1
        bbox = [x, y, x+w, y+h, 0, 0, imgNum, 0, 0]
        cv2.imwrite(folder + f'{frame:08d}' + ".jpg", blank)
        bboxes.append(bbox)
    x+=1
    while y > 0:
        blank[:, :] = 255
        y -= 1
        for i in range(h):
            for j in range(w):
                blank[y +i, x + j] = roi[i,j]
        imgNum = imgNum + 1
        frame = frame + 1
        bbox = [x, y, x+w, y+h, 0, 0, imgNum, 0, 0]
        cv2.imwrite(folder + f'{frame:08d}' + ".jpg", blank)
        bboxes.append(bbox)
    while x + w < width:
        blank[:, :] = 255
        x += 1
        for i in range(h):
            for j in range(w):
                blank[y +i, x + j] = roi[i,j]
        imgNum = imgNum + 1
        frame = frame + 1
        bbox = [x, y, x+w, y+h, 0, 0, imgNum, 0, 0]
        cv2.imwrite(folder + f'{frame:08d}' + ".jpg", blank)
        bboxes.append(bbox)
    while y + h < height:
        blank[:, :] = 255
        y += 1
        for i in range(h):
            for j in range(w):
                blank[y +i, x + j] = roi[i,j]
        imgNum = imgNum + 1
        frame = frame + 1
        bbox = [x, y, x+w, y+h, 0, 0, imgNum, 0, 0]
        cv2.imwrite(folder + f'{frame:08d}' + ".jpg", blank)
        bboxes.append(bbox)
    bboxes = np.array(bboxes)
    np.save(folder+"/labelsNew.npy", bboxes)
def func2():
    pass
# func()
def func2():
    folder = "C:/Users/ABC/Downloads/images/"
    jpg_files = [f for f in os.listdir(folder) if f.endswith(".jpg")]
    jpg_files.sort()

    bboxes = np.load("C:/Users/ABC/Downloads/images/labels.npy")
    if len(jpg_files) == bboxes.shape[0]:
        print("OK")
    else: exit
    for i,jpg in enumerate(jpg_files):
        img = cv2.imread(folder + jpg)
        # print(type(bboxes[i][0]))
        x_min, y_min, x_max, y_max = int(bboxes[i][0]), int(bboxes[i][1]), int(bboxes[i][2]), int(bboxes[i][3])
        img1 = cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255,0,0),2)
    # print(bboxes)

        
        cv2.imwrite("C:/Users/ABC/Desktop/imageWithBox/" + jpg, img1)    
# print(arr)
func2()