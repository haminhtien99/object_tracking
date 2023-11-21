import cv2
def take_roi(path_to_img):
    img = cv2.imread(path_to_img)
    print(img.shape[0], img.shape[1])
    bbox = cv2.selectROI(img, False)
    x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
    roi = img[y: y+h,x:x+w]
    # roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    cv2.destroyAllWindows()
    return roi, bbox
path = "C:/Users/ABC/Downloads/images/00000455.jpg"







_, bbox = take_roi(path)
print(bbox)