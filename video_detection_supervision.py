import cv2 as cv
import os
from ultralytics import YOLO
import supervision as sv

model = YOLO("last.pt")  # load a custom model

PATH = os.getcwd()

VIDEO_PATH = os.path.join(PATH,"videos","boat2.mp4") 

vid = cv.VideoCapture(VIDEO_PATH)

box_annotator = sv.BoxAnnotator(
    thickness=2,
)

label_annotator = sv.LabelAnnotator()

while True:

    ret,frame = vid.read()
    
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)

    labels = [
        f"{data['class_name'][class_id]} {confidence:0.2f}"
        for _,_,confidence,class_id, _,data
        in detections
    ]
    # print(detections)
    frame = box_annotator.annotate(
        scene=frame.copy(), detections=detections)
    frame = label_annotator.annotate(
        frame, detections=detections, labels=labels)
    
    cv.imshow("frame",frame)
    if cv.waitKey(1) & 0xFF == ord('q'): 
        break


vid.release() 
# Destroy all the windows 
cv.destroyAllWindows() 
