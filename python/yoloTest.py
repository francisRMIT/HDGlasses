import cv2
import time
import keyboard
from ultralytics import YOLO

#capture = cv2.VideoCapture(0)
model = YOLO("yolo26n.pt")
#i = 0
results = model.predict(source=0, show=True)

#def YOLOingIt():
        #ret, frame = capture.read()
        #results = model.predict(frame)
        #for r in results:
        #        boxes = r.boxes
        #for box in boxes:
        #        box

 


