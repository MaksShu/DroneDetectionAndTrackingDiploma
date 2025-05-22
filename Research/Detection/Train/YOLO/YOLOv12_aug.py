from ultralytics import YOLO
import torch


model = YOLO("yolo12n.yaml")

model.train(
    data="/kaggle/working/data.yaml",
    epochs=50,
    imgsz=640,
    name="augment1",
    batch=16,
    workers=2,
    close_mosaic=10,
    # augmentations
    auto_augment=None,
    hsv_h = 0.015, hsv_s = 0.7, hsv_v = 0.4, degrees = 5.0, translate = 0.2, scale = 0, shear = 0.05, perspective = 0.0005, flipud = 0.0, fliplr = 0.5, mosaic = 1.0, mixup = 0.2, copy_paste = 0.0
)