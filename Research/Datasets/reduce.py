import os
import cv2
import numpy as np
from tqdm import tqdm

ROOT_DIR = './'
SPLITS = ['train', 'test']

# Thresholds
MOVE_THRESH_RATIO = 0.05
SIZE_THRESH_RATIO = 0.05

def parse_label(label_path):
    if not os.path.exists(label_path):
        return None
    with open(label_path, 'r') as f:
        lines = f.readlines()
    if not lines:
        return None
    parts = list(map(float, lines[0].strip().split()))
    return parts[1:]

def filter_split(split):
    img_dir = os.path.join(ROOT_DIR, 'images', split)
    lbl_dir = os.path.join(ROOT_DIR, 'labels', split)

    images = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])

    if not images:
        return

    # Get image size from first image
    sample_img = cv2.imread(os.path.join(img_dir, images[0]))
    h, w = sample_img.shape[:2]
    diag = (w**2 + h**2) ** 0.5
    move_thresh = MOVE_THRESH_RATIO * diag

    prev_bbox = None

    for img_name in tqdm(images):
        lbl_name = img_name.replace('.jpg', '.txt')
        lbl_path = os.path.join(lbl_dir, lbl_name)

        bbox = parse_label(lbl_path)

        if bbox is None:
            continue

        x_c, y_c, bw, bh = bbox
        abs_x = x_c * w
        abs_y = y_c * h
        area = bw * bh * w * h

        if prev_bbox is None:
            prev_bbox = (abs_x, abs_y, area)
            continue

        # Calculate movement and size change
        move_dist = ((abs_x - prev_bbox[0])**2 + (abs_y - prev_bbox[1])**2) ** 0.5
        size_change = abs(area - prev_bbox[2])

        size_thresh = SIZE_THRESH_RATIO * prev_bbox[2]

        if move_dist < move_thresh and size_change < size_thresh:
            # Remove redundant frame
            os.remove(os.path.join(img_dir, img_name))
            os.remove(lbl_path)
        else:
            prev_bbox = (abs_x, abs_y, area)

    print(f"✅ Filtered {split} set")

for split in SPLITS:
    filter_split(split)

print("🎯 Dataset reduction completed!")