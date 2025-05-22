import os
import glob
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import cv2
import albumentations as A
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch import ToTensorV2
from torchvision.ops import box_iou
from tqdm import tqdm
from torchsummary import summary
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"


class CustomDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None, train=True):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.train = train
        self.image_paths = []
        self.class_counts = {}

        all_images = glob.glob(os.path.join(image_dir, "*.jpg"))
        for img_path in all_images:
            label_path = os.path.join(label_dir, os.path.basename(img_path).replace(".jpg", ".txt"))
            if not os.path.exists(label_path):
                if self.train:
                    continue
                self.image_paths.append(img_path)
                continue

            with open(label_path, "r") as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]

            if self.train and not lines:
                continue

            self.image_paths.append(img_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label_path = os.path.join(self.label_dir, os.path.basename(img_path).replace(".jpg", ".txt"))

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape

        boxes = []
        labels = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    data = line.strip().split()
                    if not data:
                        continue
                    class_id = int(data[0])
                    x_center, y_center, w_obj, h_obj = map(float, data[1:5])

                    x_min = (x_center - w_obj / 2) * w
                    y_min = (y_center - h_obj / 2) * h
                    x_max = (x_center + w_obj / 2) * w
                    y_max = (y_center + h_obj / 2) * h

                    boxes.append([x_min, y_min, x_max, y_max])
                    labels.append(class_id)

        if self.train and boxes:
            areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
            main_idx = np.argmax(areas)
            boxes = [boxes[main_idx]]
            labels = [labels[main_idx]]

        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4))
        labels = torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,))

        if self.transform:
            transformed = self.transform(
                image=image,
                bboxes=boxes.tolist(),
                labels=labels.tolist(),
            )
            image = torch.tensor(transformed["image"], dtype=torch.float32)
            boxes = torch.tensor(transformed["bboxes"], dtype=torch.float32)
            labels = torch.tensor(transformed["labels"], dtype=torch.int64)

        return image, boxes, labels


transform = A.Compose([
    A.Resize(480, 640),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    ToTensorV2()
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))


def collate_fn(batch):
    images, boxes, labels = zip(*batch)
    images = torch.stack(images)
    return images, boxes, labels


train_dataset = CustomDataset(
    "../../Datasets/Detection Dataset/train/images",
    "../../Datasets/Detection Dataset/train/labels",
    transform=transform,
    train=True
)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn, pin_memory=True)

val_dataset = CustomDataset(
    "../../Datasets/Detection Dataset/test/images",
    "../../Datasets/Detection Dataset/test/labels",
    transform=transform,
    train=False
)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn, pin_memory=True)


class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv4 = nn.Conv2d(32, 48, 3, padding=1)
        self.conv5 = nn.Conv2d(48, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 20 * 20, 512)
        self.fc_bbox = nn.Linear(512, 4)
        self.fc_class = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc_bbox(x), self.fc_class(x)


patience = 10
num_classes = 2
model = SimpleCNN(num_classes).to(device)

summary(model, input_size=(3, 640, 640), device="cuda")

optimizer = optim.Adam(model.parameters(), lr=0.0001)
bbox_criterion = nn.SmoothL1Loss().to(device)
cls_criterion = nn.CrossEntropyLoss().to(device)


def calculate_metrics(pred_boxes, pred_classes, true_boxes, true_labels, conf_thresh=0.5, iou_thresh=0.5):
    tp, fp, fn = 0, 0, 0
    confidences = F.softmax(pred_classes, dim=1).max(dim=1)[0]

    for i in range(len(pred_boxes)):
        if confidences[i] < conf_thresh:
            fn += len(true_boxes[i])
            continue

        if len(true_boxes[i]) == 0:
            fp += 1
            continue

        ious = box_iou(pred_boxes[i].unsqueeze(0), true_boxes[i])
        max_iou, best_idx = torch.max(ious, dim=1)

        cls_match = (pred_classes[i].argmax() == true_labels[i][best_idx])

        if max_iou >= iou_thresh and cls_match:
            tp += 1
            fn += len(true_boxes[i]) - 1
        else:
            fp += 1
            fn += len(true_boxes[i])

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    return precision, recall


train_losses = []
val_losses = []
precisions = []
recalls = []
maps = []

epochs = 100
best_map = 0.0

with open('training_log.csv', 'w') as f:
    f.write('epoch,train_loss,val_loss,precision,recall,map,best_map,early_stop_counter\n')

for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for images, boxes, labels in tqdm(train_loader):
        images = images.to(device, non_blocking=True)
        boxes = torch.cat([b.to(device, non_blocking=True) for b in boxes])
        labels = torch.cat([l.to(device, non_blocking=True) for l in labels])

        optimizer.zero_grad()
        pred_boxes, pred_classes = model(images)

        loss_bbox = bbox_criterion(pred_boxes, boxes)
        loss_cls = cls_criterion(pred_classes, labels)
        loss = loss_bbox + loss_cls

        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0.0
    all_pred_boxes = []
    all_pred_classes = []
    all_true_boxes = []
    all_true_labels = []

    with torch.no_grad():
        for images, boxes, labels in val_loader:
            images = images.to(device, non_blocking=True)
            pred_boxes, pred_classes = model(images)

            all_pred_boxes.extend(pred_boxes)
            all_pred_classes.extend(pred_classes)
            all_true_boxes.extend([b.to(device) for b in boxes])
            all_true_labels.extend([l.to(device) for l in labels])

            main_boxes = []
            main_labels = []
            for b, l in zip(boxes, labels):
                if len(b) > 0:
                    main_boxes.append(b[0])
                    main_labels.append(l[0])
                else:
                    main_boxes.append(torch.zeros(4, dtype=torch.float32))
                    main_labels.append(torch.tensor(0, dtype=torch.int64))
            main_boxes = torch.stack(main_boxes).to(device)
            main_labels = torch.stack(main_labels).to(device)

            loss_bbox = bbox_criterion(pred_boxes, main_boxes)
            loss_cls = cls_criterion(pred_classes, main_labels)
            val_loss += (loss_bbox + loss_cls).item()

    precision, recall = calculate_metrics(
        torch.stack(all_pred_boxes),
        torch.stack(all_pred_classes),
        all_true_boxes,
        all_true_labels
    )
    map_score = (precision + recall) / 2

    train_loss_epoch = train_loss / len(train_loader)
    val_loss_epoch = val_loss / len(val_loader)

    print(f"Epoch {epoch + 1}")
    print(f"Train Loss: {train_loss_epoch:.4f}")
    print(f"Val Loss: {val_loss_epoch:.4f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, mAP: {map_score:.4f}")

    train_losses.append(train_loss_epoch)
    val_losses.append(val_loss_epoch)
    precisions.append(precision)
    recalls.append(recall)
    maps.append(map_score)

    if map_score > best_map:
        best_map = map_score
        torch.save(model.state_dict(), "best_model.pth")
        print("Saved best model!")
        early_stop_counter = 0
    else:
        early_stop_counter += 1

    with open('training_log.csv', 'a') as f:
        f.write(
            f"{epoch + 1},{train_loss_epoch:.4f},{val_loss_epoch:.4f},"
            f"{precision:.4f},{recall:.4f},{map_score:.4f},"
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
