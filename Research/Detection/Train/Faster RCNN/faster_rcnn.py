import numpy as np
import torchmetrics
from matplotlib import pyplot as plt
import torch.optim as optim
from tqdm import tqdm
from torchvision.models import vgg16
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision import transforms
from roboflow import Roboflow
from torch.utils.data import DataLoader
import os
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import json
import os
from PIL import Image

transform = transforms.Compose([
    transforms.ToTensor(),
])

dataset_path = os.path.normpath("/kaggle/input/drone-detection")


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, image_path, annotations_path, transforms=None):

        with open(annotations_path, "r") as f:
            annotations = json.load(f)

        image_paths = [os.path.join(image_path, img["file_name"]) for img in annotations["images"]]

        image_id_to_annots = {img["id"]: [] for img in annotations["images"]}
        for ann in annotations["annotations"]:
            image_id_to_annots[ann["image_id"]].append(ann)

        self.image_paths = image_paths
        self.annotations = image_id_to_annots
        self.transforms = transforms

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image_id = idx

        annots = self.annotations.get(image_id, [])
        boxes, labels = [], []

        for obj in annots:
            x_min, y_min, width, height = obj["bbox"]
            x_max = x_min + width
            y_max = y_min + height
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(obj["category_id"])

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }

        if self.transforms:
            image = self.transforms(image)

        return image, target

    def __len__(self):
        return len(self.image_paths)


train_dataset = CustomDataset(f"{dataset_path}/train", f"{dataset_path}/train/_annotations.coco.json", transforms=transform)
val_dataset = CustomDataset(f"{dataset_path}/valid", f"{dataset_path}/valid/_annotations.coco.json",transform=transform)


device = "cuda" if torch.cuda.is_available() else "cpu"

patience = 10
num_classes = 2
batch_size = 4

vgg_backbone = vgg16(pretrained=True).features
backbone = torch.nn.Sequential(*list(vgg_backbone.children())[:-1])

vgg = vgg16(pretrained=True)

backbone = vgg.features

backbone.out_channels = 512

anchor_generator = AnchorGenerator(
    sizes=((32, 64, 128, 256, 512),),
    aspect_ratios=((0.5, 1.0, 2.0),) * 5
)

model = FasterRCNN(
    backbone,
    num_classes=num_classes,
    rpn_anchor_generator=anchor_generator
)

model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.0001)


def collate_fn(batch):
    images = []
    targets = []

    print(batch)

    for img, target in batch:
        images.append(img.to(device))
        target = {key: torch.as_tensor(value).to(device) for key, value in target[0].items()}
        targets.append(target)

    return images, targets


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

sample_image, sample_target = train_dataset[0]
print("Sample Target:", sample_target)


def calculate_map(model, val_loader, device, iou_threshold=0.5):
    model.eval()
    metric = torchmetrics.detection.MeanAveragePrecision(iou_thresholds=[iou_threshold]).to(device)

    with torch.no_grad():
        for images, targets in val_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images)

            preds = []
            gts = []
            for i in range(len(images)):
                pred_boxes = outputs[i]["boxes"].detach().cpu()
                pred_scores = outputs[i]["scores"].detach().cpu()
                pred_labels = outputs[i]["labels"].detach().cpu()

                gt_boxes = targets[i]["boxes"].detach().cpu()
                gt_labels = targets[i]["labels"].detach().cpu()

                preds.append({"boxes": pred_boxes, "scores": pred_scores, "labels": pred_labels})
                gts.append({"boxes": gt_boxes, "labels": gt_labels})

            metric.update(preds, gts)

    return metric.compute()["map"].item()


train_losses = []
val_losses = []
precisions = []
recalls = []
maps = []

epochs = 100
best_map = 0.0

with open('training_log.csv', 'w') as f:
    f.write('epoch,train_loss,map,best_map,early_stop_counter,val_loss\n')

for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for images, targets in tqdm(train_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, targets in tqdm(val_loader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())
            val_loss += loss.item()

    map_score = calculate_map(model, val_loader, device)

    train_loss_epoch = train_loss / len(train_loader)
    val_loss_epoch = val_loss / len(val_loader)

    print(f"Epoch {epoch + 1}")
    print(f"Train Loss: {train_loss_epoch:.4f}")
    print(f"Val Loss: {val_loss_epoch:.4f}")
    print(f"mAP: {map_score:.4f}")

    train_losses.append(train_loss_epoch)
    val_losses.append(val_loss_epoch)
    maps.append(map_score)

    if map_score > best_map:
        best_map = map_score
        torch.save(model.state_dict(), "fasterrcnn_vgg16_best.pth")
        print("Saved best model!")
        early_stop_counter = 0
    else:
        early_stop_counter += 1

    with open('training_log.csv', 'a') as f:
        f.write(
            f"{epoch + 1},{train_loss_epoch:.4f},{val_loss_epoch:.4f},"
            f"{map_score:.4f},"
            f"{best_map:.4f},{early_stop_counter}\n"
        )

    if early_stop_counter >= patience:
        print("Early stopping triggered. Training stopped.")
        break


def normalize(y):
    return (y - np.min(y)) / (np.max(y) - np.min(y))


train_loss_norm = normalize(train_losses)
val_loss_norm = normalize(val_losses)

plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss Curves')
plt.savefig('loss_curves.png')
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(precisions, label='Precision')
plt.plot(recalls, label='Recall')
plt.plot(maps, label='mAP')
plt.xlabel('Epochs')
plt.ylabel('Value')
plt.legend()
plt.title('Precision, Recall, and mAP Curves')
plt.savefig('metrics_curves.png')
plt.close()
