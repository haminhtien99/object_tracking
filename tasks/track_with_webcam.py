import cv2



def webcam(tracker = None):
    capture = cv2.VideoCapture(0)
    if not  capture.isOpened():
        print("cannot open the video capture")
    running = True
    tracking = False
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
                bbox = [bbox[0], bbox[1],bbox[0]  + bbox[2],bbox[1]+ bbox[3]]
                Roi = clone[bbox[1]: bbox[3], bbox[0]:bbox[2]]
                print(f'[x1, y1, x2, y2] [{bbox[0]}, {bbox[1]}, {bbox[0] + bbox[2]}, {bbox[1] + bbox[3]}]')
                if tracker == None:
                    cv2.imshow('roi', Roi)
                else: 
                    tracking = True
                    clone = clone[:,:, ::-1] #convert BGR to RGB
                    print('tracking')
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0,0), 2)
                    cv2.imshow('track window', frame)
                    tracker.track('object', clone, bbox)    
                    
                
                break
        if tracker is not None and tracking is True:
            clone =  frame.copy()
            clone = clone[:,:, ::-1]
            bbox = tracker.track('object', clone)
            bbox = [int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])]
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0,0), 2)
            cv2.imshow('track window', frame)
        # Close program with keyboard Esc
        if key == 27:
            capture.release()
            cv2.destroyAllWindows()
            running = False

if __name__ == "__main__":
    webcam()