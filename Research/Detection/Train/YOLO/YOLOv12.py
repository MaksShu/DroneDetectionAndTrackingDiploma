from ultralytics import YOLO
import torch


model = YOLO("yolo12n.yaml")

model.train(
    data="/kaggle/working/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
)