import os
import json

dataset_path = "../Datasets/Detection Dataset/"
output_path = "../Datasets/Detection Dataset/yolo_labels/"

os.makedirs(output_path, exist_ok=True)

for split in ["train", "test"]:
    json_dir = os.path.join(dataset_path, split)
    label_dir = os.path.join(output_path, split)
    os.makedirs(label_dir, exist_ok=True)

    for file in os.listdir(json_dir):
        if file.endswith(".json"):
            json_path = os.path.join(json_dir, file)

            with open(json_path, "r") as f:
                data = json.load(f)

            img_w, img_h = data["width"], data["height"]
            img_name = os.path.splitext(data["filename"])[0]

            label_txt = os.path.join(label_dir, img_name + ".txt")
            with open(label_txt, "w") as f_out:
                if "bndboxes" in data:
                    for box in data["bndboxes"]:
                        x_min, y_min, x_max, y_max = box["xmin"], box["ymin"], box["xmax"], box["ymax"]

                        x_center = (x_min + x_max) / 2.0 / img_w
                        y_center = (y_min + y_max) / 2.0 / img_h
                        width = (x_max - x_min) / img_w
                        height = (y_max - y_min) / img_h

                        f_out.write(f"0 {x_center} {y_center} {width} {height}\n")
