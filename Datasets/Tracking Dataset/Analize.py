import json
import cv2
import os

import pandas as pd

from utils import plot_stats

folder = 'test'

if folder == 'train':
    name = 'Тренувальна вибірка'
elif folder == 'test':
    name = 'Тестова вибірка'
else:
    name = 'Валідаційна вибірка'

video_files = [f for f in os.listdir(folder) if f.endswith('.mp4')]

frame_sizes = []
drone_data = []
drone_scale_ratios = []
drone_positions = []
drone_aspect_ratios = []


def get_data(video_path):
    json_path = video_path.split('.')[0] + '.json';

    with open(json_path, 'r') as f:
        frame_data = json.load(f)

    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    for frame_idx, data in enumerate(frame_data['exist']):
        exist = data

        frame_sizes.append({'width': frame_width, 'height': frame_height})

        if exist == 1:
            bbox = frame_data['gt_rect'][frame_idx]
            if bbox == [] or bbox == [0, 0, 0, 0]:
                continue

            x, y, width, height = bbox

            if height == 0:
                continue

            scale_ratio = width / (frame_width)
            aspect_ratio = width / height
            position = (x, y)

            drone_positions.append(position)
            drone_scale_ratios.append(scale_ratio)
            drone_aspect_ratios.append(aspect_ratio)

    cap.release()


for video_file in video_files:
    get_data(os.path.join(folder, video_file))


frame_df = pd.DataFrame(frame_sizes)
print(f'Total frames count: {frame_df.size}')


drone_df = pd.DataFrame({
    "drone_center_x": [loc[0] for loc in drone_positions],
    "drone_center_y": [loc[1] for loc in drone_positions],
    "scaled_drone_size": drone_scale_ratios,
    "drone_aspect_ratio": drone_aspect_ratios
})

plot_stats(drone_df, name)

