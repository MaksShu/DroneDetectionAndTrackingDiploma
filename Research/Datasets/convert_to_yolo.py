import os
import json
import cv2
from tqdm import tqdm

ROOT_DIR = './'
OUT_DIR = '../YOLO Detection Dataset 2'
SPLITS = ['train', 'val', 'test']


def process_split(split):
    input_dir = os.path.join(ROOT_DIR, split)
    images_dir = os.path.join(OUT_DIR, 'images', split)
    labels_dir = os.path.join(OUT_DIR, 'labels', split)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    files = [f for f in os.listdir(input_dir) if f.endswith('.mp4')]

    for video_file in tqdm(files):
        base_name = video_file[:-4]
        json_path = os.path.join(input_dir, base_name + '.json')
        video_path = os.path.join(input_dir, video_file)

        with open(json_path, 'r') as f:
            annotations = json.load(f)

        exists = annotations['exist']
        gt_rects = annotations['gt_rect']

        cap = cv2.VideoCapture(video_path)
        frame_idx = 0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_name = f"{base_name}_frame_{frame_idx:05d}.jpg"
            label_name = f"{base_name}_frame_{frame_idx:05d}.txt"

            cv2.imwrite(os.path.join(images_dir, frame_name), frame)

            label_path = os.path.join(labels_dir, label_name)
            with open(label_path, 'w') as lf:
                if exists[frame_idx] == 1:
                    bbox = gt_rects[frame_idx]
                    x_center = (bbox[0] + bbox[2] / 2) / width
                    y_center = (bbox[1] + bbox[3] / 2) / height
                    w_norm = bbox[2] / width
                    h_norm = bbox[3] / height
                    lf.write(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

            frame_idx += 1

        cap.release()


for split in SPLITS:
    process_split(split)

print("Conversion completed!")