from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8s.pt")

    model.train(data="dataset.yaml",
                epochs=100,
                batch=16,
                imgsz=640,
                optimizer="AdamW",
                patience=10,
                half=True,
                verbose=True
                )
