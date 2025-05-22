import torch
import torchvision
import cv2
import time
from ultralytics import YOLO

model = YOLO("/kaggle/input/150-weight/last.pt")
model.model.fuse()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

model.eval()

video_path = "/kaggle/input/test-yolo-set/test2.mp4"
cap = cv2.VideoCapture(video_path)

num_frames = 0
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    with torch.no_grad():
        outputs = model.predict(frame)

    num_frames += 1

elapsed_time = time.time() - start_time
fps = num_frames / elapsed_time

print(f"FPS: {fps:.2f} frames per second")

cap.release()
cv2.destroyAllWindows()