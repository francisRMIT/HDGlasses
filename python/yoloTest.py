import cv2
import time
import keyboard
from ultralytics import YOLO

capture = cv2.VideoCapture(0)
model = YOLO("yolo26n.pt")


def YOLOingIt():
        ret, frame = capture.read()
        results = model.predict(frame, verbose=False)
        for r in results:
                cv2.imshow("camera", r.plot())
                cv2.waitKey(1)
                boxes = r.boxes
        for box in boxes:
                clsid = box.cls.int()
                conf = round(box.conf.item(),2)
                name = r.names[clsid.item()]
                print(name, " ", conf)

 

while True:
        if keyboard.is_pressed("space"):
                cv2.destroyAllWindows()
                YOLOingIt()
                time.sleep(0.5)

        if keyboard.is_pressed("q"):
                cv2.destroyAllWindows()
                time.sleep(0.5)

        if keyboard.is_pressed("esc"):
                break