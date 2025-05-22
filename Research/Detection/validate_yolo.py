from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("best.pt")

    res = model.val(data="eval.yaml")
    print(res)
