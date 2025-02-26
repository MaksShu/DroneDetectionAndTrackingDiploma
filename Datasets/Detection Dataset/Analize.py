import os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import plot_stats

output_folder = 'test'

if output_folder == 'train':
    name = 'Тренувальна вибірка'
else:
    name = 'Тестова вибірка'

scaled_drone_sizes = []
aspect_ratios = []
drone_locations = []

json_files = [f for f in os.listdir(output_folder) if f.endswith('.json')]

for json_file in json_files:
    with open(os.path.join(output_folder, json_file), 'r') as file:
        data = json.load(file)

        image_width = data["width"]
        image_height = data["height"]

        if data["drone"]:
            for bndbox in data["bndboxes"]:
                center_x = (bndbox["xmin"] + bndbox["xmax"]) // 2
                center_y = (bndbox["ymin"] + bndbox["ymax"]) // 2
                drone_locations.append((center_x, center_y))

                drone_width = bndbox["xmax"] - bndbox["xmin"]
                drone_height = bndbox["ymax"] - bndbox["ymin"]
                scaled_width = drone_width / image_width
                scaled_height = drone_height / image_height
                scaled_drone_sizes.append(scaled_width)

                drone_aspect_ratio = drone_width / drone_height
                aspect_ratios.append(drone_aspect_ratio)

df = pd.DataFrame({
    "drone_center_x": [loc[0] for loc in drone_locations],
    "drone_center_y": [loc[1] for loc in drone_locations],
    "scaled_drone_size": scaled_drone_sizes,
    "drone_aspect_ratio": aspect_ratios,
})

plot_stats(df, name)
