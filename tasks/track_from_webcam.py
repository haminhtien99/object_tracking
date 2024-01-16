import cv2
import os
import sys
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from modules.webcamera import staticROI


class track_staticROI(staticROI):
    def __init__(self):
        super().__init__()
        self.tracker = None
    def update(self, track = None):
        self.tracker = track
        if not self.capture.isOpened():
            print("cannot open the video capture")
            return
        running = True
        tracking = False
        while running:
            # Read frame
            (self.status, self.frame) = self.capture.read()
            if tracking is False:
                cv2.imshow('image', self.frame)
                key = cv2.waitKey(2)

            # Crop image
            if key == ord('c'):
                tracking = True
                self.clone = self.frame.copy()
                cv2.namedWindow('image')
                cv2.setMouseCallback('image', self.extract_coordinates)
                while True:
                    key = cv2.waitKey(2)
                    cv2.imshow('image', self.clone)
                    
                    if key == ord('\r'): # keyboard Enter
                        # Crop and display cropped image
                        self.crop_ROI()
                        self.show_cropped_ROI()
                        print('tracking ....')
                        # Resume video
                        break
            
            if tracking is True:
                self.frame = cv2.rectangle(self.frame, self.image_coordinates[0], self.image_coordinates[1], (0,255, 0), 2)
                cv2.imshow('image', self.frame)
                key = cv2.waitKey(2)
            # Close program with keyboard Esc
            if key == 27:
                self.capture.release()
                cv2.destroyAllWindows()
                running = False
                
                
track = track_staticROI()
track.update()