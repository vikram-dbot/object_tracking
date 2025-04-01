from ultralytics import YOLO
import cv2

model = YOLO("best.pt")
result = model(rf"C:\Users\Welcome\phases_detection\output\video25\video25_18.jpg",save=True)
result_json = result[0].tojson()
print(result_json)