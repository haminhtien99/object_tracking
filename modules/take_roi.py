import cv2
def take_roi(path_to_img):
    img = cv2.imread(path_to_img)
    imgResize = cv2.resize(img, (1000,1000))
    print(img.shape[0], img.shape[1])
    bbox = cv2.selectROI(imgResize, False)
    # x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
    x = int(bbox[0] * img.shape[1]/1000)
    y = int(bbox[1] * img.shape[0]/1000)
    w= int(bbox[2] * img.shape[1]/1000)
    h = int(bbox[3] * img.shape[0]/1000)
    
    roi = img[y: y+h,x:x+w]
    # roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    cv2.destroyAllWindows()
    return roi, bbox, x,y,w,h

path = "C:/Users/ABC/Downloads/Telegram Desktop/Xian_station-CN_GoogleSat_v0.tif"







if __name__ == "__main__":
    _, bbox, x,y,w,h = take_roi(path)
    print(bbox,x,y,w,h)

