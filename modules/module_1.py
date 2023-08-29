import cv2
import os
import shutil
## co le nen can class folder voi hai thuoc tinh images va list_images

def read_frame(path, fps):
    list_jpg = os.listdir(path)
    filenames = [f for f in list_jpg]
    filenames.sort()
    full_paths=[]

    for f in filenames: 
        full_paths.append(path+'/'+f)
    # print(full_paths)

    for filename in full_paths:
        # print(filename) 
        frame = cv2.imread(filename)
        cv2.imshow('frame',frame)

        #press Esc to quit
        k = cv2.waitKey(int(1000/fps)) & 0xff
        if k == 27 : break


    cv2.destroyAllWindows()

def track_frame(path, tracker):
    list_jpg = os.listdir(path)
    filenames = [f for f in list_jpg]
    filenames.sort()
    full_paths=[]
    
    for f in filenames:
        
        full_paths.append(path+'/'+f)

    # print(full_paths)
    
    first_path = full_paths[0]
    first_frame = cv2.imread(first_path)
    full_paths.remove(first_path)

    # Define an initial bounding box
    bbox = (287, 23, 86, 320)
    # Uncomment the line below to select a different bounding box
    bbox = cv2.selectROI(first_frame, False)

    # Initialize tracker with first frame and bounding box
    ok = tracker.init(first_frame, bbox)
    
    for filename in full_paths:
        # print(filename) 
        frame = cv2.imread(filename)

        # Start timer
        timer = cv2.getTickCount()

        # Update tracker
        ok, bbox = tracker.update(frame)
        
        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        
        # Draw bounding box
        if ok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
        else :
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2);
            
        # Display tracker type on frame
        tracker_type = str(tracker)
        cv2.putText(frame, tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
     
        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);
        
        # Display result
        cv2.imshow("Tracking", frame)
        #press Esc to quit
        k = cv2.waitKey(1) & 0xff
        if k == 27 : break

    cv2.destroyAllWindows()

def take_image(folders, outFolder):
    for folder in folders:
        list_jpg = os.listdir(folder)
        filenames = [f for f in list_jpg]
        jpg = folder +'/' + filenames[0]
        shutil.copy(jpg, outFolder)
def choose_video_on_demand(Folder):
    return 0