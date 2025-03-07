from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8s.pt")

    model.tune(
        data="dataset.yaml",
        epochs=100,
        iterations=100,
        imgsz=640,
        device="cuda"
    )
