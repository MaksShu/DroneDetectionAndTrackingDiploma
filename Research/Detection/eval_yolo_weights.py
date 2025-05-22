import os
from pathlib import Path

datasets = {
    "EvalDataset": {
        "path": "/kaggle/input/evaluation-dataset/Evaluating Dataset/images/test",
        "nc":   1,
        "names":["drone"]
    },
    "BackgroundDataset": {
        "path": "/kaggle/input/yolo-merged-and-backgrounds-dataset/YOLO Merged and Backgrounds Dataset/YOLO Merged and Backgrounds Dataset/images/test",
        "nc":   1,
        "names":["drone"]
    }
}

cfg_dir = Path("data_configs")
cfg_dir.mkdir(exist_ok=True)

for ds_name, ds in datasets.items():
    cfg_path = cfg_dir / f"{ds_name}.yaml"
    with open(cfg_path, "w") as f:
        f.write(f"nc: {ds['nc']}\n")
        f.write(f"names: {ds['names']}\n")
        f.write(f"train: {ds['path']}\n")
        f.write(f"val:   {ds['path']}\n")
    print(f"→ Wrote config: {cfg_path}")

from ultralytics import YOLO

model_paths = list(Path("/kaggle/input/150-best").rglob("*.pt"))

all_res = []

for model_path in model_paths:
    print(f"\n\n=== MODEL: {model_path} ===")
    model = YOLO(str(model_path))
    for ds_name in datasets:
        cfg_file = cfg_dir / f"{ds_name}.yaml"
        print(f"\n--> validating on {ds_name}")
        results = model.val(
            data    = str(cfg_file),
            batch   = 16,
            imgsz   = 640,
            verbose = False
        )

        all_res.append(results)
        
        print(f"  mAP50:    {results.box.map50:.4f}")
        print(f"  mAP50-95: {results.box.map:.4f}")