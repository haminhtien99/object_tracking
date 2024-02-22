import cv2



def webcam():
    capture = cv2.VideoCapture(0)
    if not  capture.isOpened():
        print("cannot open the video capture")
    running = True
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    while running:
    # Read frame

        (status, frame) =  capture.read()
        cv2.imshow('image',  frame)
        key = cv2.waitKey(2)

        # Crop image
        if key == ord('c'):
            clone =  frame.copy()
            cv2.namedWindow('image')
            while True:
                key = cv2.waitKey(2)
                bbox = cv2.selectROI('image', clone)
                Roi = clone[bbox[1]: bbox[1]+ bbox[3], bbox[0]:bbox[0] +bbox[2]]

                cv2.imshow('roi', Roi)
                print(f'[x1, y1, x2, y2] [{bbox[0]}, {bbox[1]}, {bbox[0] + bbox[2]}, {bbox[1] + bbox[3]}]')
                break
        # Close program with keyboard Esc
        if key == 27:
            capture.release()
            cv2.destroyAllWindows()
            running = False
if __name__ == '__main__':
    webcam()
    