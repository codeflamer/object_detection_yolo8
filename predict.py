from ultralytics import YOLO
import cv2 as cv

# Load a model

model = YOLO("last.pt")  # load a custom model

# Predict with the model
# results = model("./test/images/00b1aabfdf50a42b_jpg.rf.187558a3df0bd0be6026ab923d10532f.jpg")  # predict on an image

image = cv.imread("./test/images/1fbb1e6045c859d2_jpg.rf.c8d345bd365a5b139191a65c0276b12d.jpg")

overall_detections = []

def detect_objects(image):
    results = model(image)[0]

    detections = []
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        detections.append([int(x1), int(y1), int(x2), int(y2), round(score, 3),
                           results.names[int(class_id)]])

    return detections

for detection in detect_objects('./test/images/1fbb1e6045c859d2_jpg.rf.c8d345bd365a5b139191a65c0276b12d.jpg'): 
    overall_detections.append(detection) 

for detection in overall_detections:
    cv.rectangle(image,(detection[0],detection[1]),(detection[2],detection[3]),(255, 0, 0),2)
    cv.imshow("bounding box",image)        
cv.waitKey(0) 
cv.destroyAllWindows()             