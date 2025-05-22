import os
import glob
import cv2
import numpy as np
from ultralytics import YOLO
from tqdm.auto import tqdm

# Model and dataset paths
WEIGHTS = "/kaggle/input/150-best/best.pt"
IMAGES_DIR = "/kaggle/input/evaluation-dataset/Evaluating Dataset/images/test"
LABELS_DIR = "/kaggle/input/evaluation-dataset/Evaluating Dataset/labels/test"

# Detection parameters
IOU_THRESHOLD = 0.5
CONF_THRESHOLD = 0.1
NMS_THRESHOLD = 0.45

def convert_yolo_to_absolute(x, y, w, h, img_w, img_h):
    # Convert normalized YOLO coordinates to absolute pixel coordinates
    x_center = x * img_w
    y_center = y * img_h
    width = w * img_w
    height = h * img_h
    
    x_min = x_center - width/2
    y_min = y_center - height/2
    x_max = x_center + width/2
    y_max = y_center + height/2
    
    return [x_min, y_min, x_max, y_max]

def calculate_iou(box1, box2):
    # Calculate intersection over union for two bounding boxes
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area

    return intersection_area / union_area if union_area > 0 else 0

model = YOLO(WEIGHTS)

tp = 0
fp = 0
fn = 0

# Process each image
for img_path in tqdm(glob.glob(os.path.join(IMAGES_DIR, "*.*"))):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    # Get model predictions
    results = model(img, conf=CONF_THRESHOLD, iou=NMS_THRESHOLD, verbose=False)[0]
    preds = []
    for box, conf, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
        preds.append({
            "xyxy": box.tolist(),
            "conf": float(conf),
            "cls": int(cls),
            "matched": False
        })

    # Load ground truth labels
    label_file = os.path.join(LABELS_DIR, os.path.splitext(os.path.basename(img_path))[0] + ".txt")
    gts = []
    if os.path.exists(label_file):
        for line in open(label_file):
            cls, x, y, bw, bh = map(float, line.split())
            gts.append({
                "xyxy": xywh2xyxy(x, y, bw, bh, w, h),
                "cls": int(cls),
                "matched": False
            })

    preds = sorted(preds, key=lambda x: x["conf"], reverse=True)
    for p in preds:
        best_iou = 0
        best_gt = None
        for g in gts:
            if g["matched"] or g["cls"] != p["cls"]:
                continue
            iou = compute_iou(p["xyxy"], g["xyxy"])
            if iou > best_iou:
                best_iou = iou
                best_gt = g
        if best_iou >= IOU_THRESHOLD:
            tp += 1
            p["matched"] = True
            best_gt["matched"] = True
        else:
            fp += 1

    for g in gts:
        if not g["matched"]:
            fn += 1

print(f"True Positives:  {tp}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")