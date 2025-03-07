import torch
import torchvision
import cv2
import os
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from torchmetrics.detection import MeanAveragePrecision
from tqdm import tqdm

class_names = ["drone"]
num_classes = len(class_names)
dataset_path = "../../Datasets/Detection Dataset/"
input_size = (480, 640)
batch_size = 8
num_epochs = 100
patience = 10
best_map = -1.0
early_stop_counter = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CustomDataset(Dataset):
    def __init__(self, root_dir, img_size=input_size, transform=None):
        self.root = root_dir
        self.img_files = [f for f in os.listdir(os.path.join(root_dir, "images"))
                          if f.endswith((".jpg", ".png"))]
        self.label_files = [f.replace(os.path.splitext(f)[1], ".txt") for f in self.img_files]
        self.img_size = img_size
        self.transform = transform or v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize(img_size, antialias=True),
        ])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "images", self.img_files[idx])
        label_path = os.path.join(self.root, "labels", self.label_files[idx])

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img.shape[:2]

        boxes = []
        labels = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    class_id, xc, yc, w, h = map(float, parts)

                    x_min = (xc - w / 2) * orig_w
                    y_min = (yc - h / 2) * orig_h
                    x_max = (xc + w / 2) * orig_w
                    y_max = (yc + h / 2) * orig_h
                    boxes.append([x_min, y_min, x_max, y_max])
                    labels.append(int(class_id))

        img = self.transform(img)
        _, new_h, new_w = img.shape

        scale_w = new_w / orig_w
        scale_h = new_h / orig_h
        boxes = torch.tensor(boxes, dtype=torch.float32)
        if len(boxes) > 0:
            boxes[:, [0, 2]] *= scale_w
            boxes[:, [1, 3]] *= scale_h

        return img, {"boxes": boxes, "labels": torch.tensor(labels, dtype=torch.int64)}


class RTDETRTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_encoder_layers=6, num_decoder_layers=6):
        super().__init__()
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=2048,
            dropout=0.1,
            activation="relu",
            batch_first=True
        )

    def forward(self, src, mask=None):
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(0, 2, 1)

        memory = self.transformer.encoder(src)
        hs = self.transformer.decoder(memory)

        return hs.permute(0, 2, 1).reshape(bs, c, h, w)


class RTDETR(nn.Module):
    def __init__(self, num_classes=80, hidden_dim=256):
        super().__init__()
        self.backbone = torchvision.models.resnet101(pretrained=True)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])

        self.transformer = RTDETRTransformer(
            d_model=hidden_dim,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6
        )

        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.input_proj = nn.Conv2d(2048, hidden_dim, kernel_size=1)

    def forward(self, x):
        features = self.backbone(x)
        features = self.input_proj(features)

        hs = self.transformer(features)

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()

        return {"pred_logits": outputs_class, "pred_boxes": outputs_coord}


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        layers = []
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(input_dim, hidden_dim), nn.ReLU()])
            input_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class RTDETRLoss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.bbox_loss = nn.L1Loss()
        self.giou_loss = torchvision.ops.generalized_box_iou_loss
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, outputs, targets):
        src_logits = outputs["pred_logits"]
        src_boxes = outputs["pred_boxes"]

        cost_class = -src_logits.softmax(-1)[..., :-1]
        cost_bbox = self.bbox_loss(src_boxes, targets["boxes"])
        cost_giou = self.giou_loss(src_boxes, targets["boxes"])

        loss = cost_class.mean() + cost_bbox.mean() + cost_giou.mean()
        return loss


model = RTDETR(num_classes=num_classes).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = RTDETRLoss(num_classes)

train_dataset = CustomDataset(os.path.join(dataset_path, "test"), input_size)
val_dataset = CustomDataset(os.path.join(dataset_path, "test"), input_size)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: x)
val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=lambda x: x)


for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader):
        images = [item[0].to(device) for item in batch]
        targets = [item[1] for item in batch]

        outputs = model(torch.stack(images))
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    model.eval()
    metric = MeanAveragePrecision()

    with torch.no_grad():
        for batch in val_loader:
            images = [item[0].to(device) for item in batch]
            targets = [item[1] for item in batch]

            outputs = model(torch.stack(images))

            preds = [{
                "boxes": outputs["pred_boxes"],
                "scores": outputs["pred_logits"].softmax(-1)[..., :-1].max(-1).values,
                "labels": outputs["pred_logits"].argmax(-1)
            }]

            metric.update(preds, targets)

    current_metrics = metric.compute()
    current_map = current_metrics['map'].item()

    if current_map > best_map:
        print(f"mAP improved from {best_map:.4f} to {current_map:.4f}. Saving model...")
        best_map = current_map
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_map': best_map,
        }, "best_rtdetr_r101.pth")
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        print(f"mAP did not improve. Patience counter: {early_stop_counter}/{patience}")

    if early_stop_counter >= patience:
        print(f"\nEarly stopping triggered after {patience} epochs without improvement!")
        break

    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    print(f"Train Loss: {total_loss / len(train_loader):.4f}")
    print(f"Validation mAP: {current_map:.4f}")
    print(f"Best mAP: {best_map:.4f}")

print("\nTraining complete!")
print(f"Best validation mAP: {best_map:.4f}")
print("Final model saved to 'best_rtdetr_r101.pth'")
