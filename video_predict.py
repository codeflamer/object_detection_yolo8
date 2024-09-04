import cv2 as cv
import os
from ultralytics import YOLO

model = YOLO("last.pt")  # load a custom model

PATH = os.getcwd()

VIDEO_PATH = os.path.join(PATH,"videos","boat1.mp4") 

vid = cv.VideoCapture(VIDEO_PATH)

ret = True

def detect_objects(image):
    results = model(image)[0]
    # print(results.boxes.data.tolist())
    detections = []
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        detections.append([int(x1), int(y1), int(x2), int(y2), round(score, 3),
                           results.names[int(class_id)]])

    return detections

while ret:

    ret,frame = vid.read()

    
    detections = detect_objects(frame)

    for detection in detections:
        cv.rectangle(frame,(detection[0],detection[1]),(detection[2],detection[3]),(255, 0, 0),2)
        print(frame)
        cv.imshow("frame",frame)

    if cv.waitKey(1) & 0xFF == ord('q'): 
        break
    break

vid.release() 
# Destroy all the windows 
cv.destroyAllWindows() 
