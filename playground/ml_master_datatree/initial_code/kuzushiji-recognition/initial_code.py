import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms as T
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import random

# Reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Parameters
MAX_IMG_SIZE = 800               # reduce from 2048
BATCH_SIZE = 1                   # batch size 1 to be safe
NUM_EPOCHS = 10
LR = 0.005
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
CONF_THRESH = 0.3
MAX_PREDS = 1200
NUM_WORKERS = 4

# Paths
TRAIN_CSV = 'input/train.csv'
SAMPLE_SUB = 'input/sample_submission.csv'
TRAIN_IMG_DIR = 'input/train_images'
TEST_IMG_DIR = 'input/test_images'
SUBMISSION_DIR = 'submission'
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Load training labels and build character mapping
df = pd.read_csv(TRAIN_CSV)
all_chars = set()
for labels in df['labels'].dropna():
    parts = labels.strip().split()
    for i in range(0, len(parts), 5):
        if i + 4 < len(parts):
            all_chars.add(parts[i])
label_to_id = {char: i+1 for i, char in enumerate(sorted(all_chars))}
id_to_char = {i+1: char for i, char in enumerate(sorted(all_chars))}
num_classes = len(label_to_id) + 1  # + background

print(f"Number of unique characters: {len(label_to_id)}")
print(f"Number of classes (including background): {num_classes}")

# Train/validation split (95%/5%)
np.random.seed(42)
msk = np.random.rand(len(df)) < 0.95
df_train = df[msk].reset_index(drop=True)
df_val = df[~msk].reset_index(drop=True)

# Dataset class
class KuzushijiDataset(Dataset):
    def __init__(self, df, img_dir, label_map, transforms=None, is_test=False, max_size=800):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.label_map = label_map
        self.transforms = transforms
        self.is_test = is_test
        self.max_size = max_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row['image_id']
        img_path = os.path.join(self.img_dir, img_id + '.jpg')
        image = cv2.imread(img_path)
        if image is None:
            image = np.zeros((512, 512, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w = image.shape[:2]
        scale = 1.0
        if max(h, w) > self.max_size:
            scale = self.max_size / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            new_w, new_h = w, h

        image = Image.fromarray(image)

        if self.transforms:
            image = self.transforms(image)

        if self.is_test:
            return image, img_id, scale

        # Parse labels for training/validation
        labels_str = row['labels']
        boxes = []
        labels = []
        if isinstance(labels_str, str):
            parts = labels_str.strip().split()
            for i in range(0, len(parts), 5):
                if i + 4 >= len(parts):
                    break
                char = parts[i]
                x = float(parts[i+1])
                y = float(parts[i+2])
                width = float(parts[i+3])
                height = float(parts[i+4])
                if width <= 0 or height <= 0:
                    continue
                x *= scale
                y *= scale
                width *= scale
                height *= scale
                x1 = x
                y1 = y
                x2 = x + width
                y2 = y + height
                # clip to image boundaries
                x1 = max(0, min(x1, new_w-1))
                y1 = max(0, min(y1, new_h-1))
                x2 = max(0, min(x2, new_w-1))
                y2 = max(0, min(y2, new_h-1))
                if x2 <= x1 or y2 <= y1:
                    continue
                boxes.append([x1, y1, x2, y2])
                if char in self.label_map:
                    labels.append(self.label_map[char])
                else:
                    continue  # should not happen
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx]),
            'area': area,
            'iscrowd': iscrowd
        }
        return image, target, img_id, scale

# Data transforms
if hasattr(T, 'GaussianBlur'):
    train_transform = T.Compose([
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.GaussianBlur(kernel_size=(3,5), sigma=(0.1, 2.0)),
        T.ToTensor()
    ])
else:
    train_transform = T.Compose([
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.ToTensor()
    ])
val_transform = T.Compose([T.ToTensor()])

# Datasets and dataloaders
train_dataset = KuzushijiDataset(df_train, TRAIN_IMG_DIR, label_to_id,
                                 transforms=train_transform, is_test=False, max_size=MAX_IMG_SIZE)
val_dataset = KuzushijiDataset(df_val, TRAIN_IMG_DIR, label_to_id,
                               transforms=val_transform, is_test=False, max_size=MAX_IMG_SIZE)

def collate_fn(batch):
    return tuple(zip(*batch))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, collate_fn=collate_fn, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, collate_fn=collate_fn, pin_memory=True)

# Model: use standard fasterrcnn_resnet50_fpn
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
    weights='COCO_V1',
    box_detections_per_img=MAX_PREDS,
    rpn_post_nms_top_n_test=2000,
    rpn_pre_nms_top_n_test=2000,
    rpn_post_nms_top_n_train=2000,
    rpn_pre_nms_top_n_train=2000
)

# Replace box predictor for our large number of classes
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Set transform sizes: multi‑scale training up to 800, fixed 800 for val/test
model.transform.min_size = (640, 672, 704, 736, 768, 800)   # random from these
model.transform.max_size = 800

model.to(device)

# Optimizer and scheduler
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

# Mixed precision scaler (new API)
scaler = torch.amp.GradScaler('cuda')

# Training loop
print("Starting training...")
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0.0
    for i, (images, targets, img_ids, scales) in enumerate(train_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += losses.item()
        if (i + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Step {i+1}, Loss: {losses.item():.4f}")

    scheduler.step()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} finished. Average loss: {avg_loss:.4f}")

print("Training completed.")

# Validation: switch to fixed size
model.eval()
model.transform.min_size = (800,)
model.transform.max_size = 800

def compute_f1(predictions, ground_truths, conf_threshold=CONF_THRESH):
    total_tp = 0
    total_pred = 0
    total_gt = 0
    for pred, gt in zip(predictions, ground_truths):
        boxes = pred['boxes'].cpu().numpy()
        scores = pred['scores'].cpu().numpy()
        labels = pred['labels'].cpu().numpy()
        keep = scores >= conf_threshold
        boxes = boxes[keep]
        labels = labels[keep]
        if len(boxes) > 0:
            order = np.argsort(-scores[keep])
            boxes = boxes[order]
            labels = labels[order]
        gt_boxes = gt['boxes'].cpu().numpy()
        gt_labels = gt['labels'].cpu().numpy()

        gt_matched = [False] * len(gt_boxes)
        if len(boxes) > 0:
            centers = (boxes[:, :2] + boxes[:, 2:]) / 2
        else:
            centers = np.zeros((0, 2))

        tp = 0
        for i in range(len(boxes)):
            cx, cy = centers[i]
            lbl = labels[i]
            for j, (gt_box, gt_lbl) in enumerate(zip(gt_boxes, gt_labels)):
                if not gt_matched[j] and gt_lbl == lbl:
                    x1, y1, x2, y2 = gt_box
                    if x1 <= cx <= x2 and y1 <= cy <= y2:
                        gt_matched[j] = True
                        tp += 1
                        break
        total_tp += tp
        total_pred += len(boxes)
        total_gt += len(gt_boxes)

    precision = total_tp / total_pred if total_pred > 0 else 0
    recall = total_tp / total_gt if total_gt > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1

all_preds = []
all_gts = []
with torch.no_grad():
    for images, targets, img_ids, scales in val_loader:
        images = [img.to(device) for img in images]
        outputs = model(images)
        all_preds.extend(outputs)
        all_gts.extend(targets)

precision, recall, f1 = compute_f1(all_preds, all_gts)
print(f"Validation Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

# Test inference and submission
print("Generating test predictions...")
df_sample = pd.read_csv(SAMPLE_SUB)
test_ids = df_sample['image_id'].tolist()
test_df = pd.DataFrame({'image_id': test_ids})
test_dataset = KuzushijiDataset(test_df, TEST_IMG_DIR, label_to_id,
                                transforms=val_transform, is_test=True, max_size=MAX_IMG_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS, collate_fn=collate_fn, pin_memory=True)

model.eval()
model.transform.min_size = (800,)
model.transform.max_size = 800
submission_dict = {}

with torch.no_grad():
    for images, img_ids, scales in test_loader:
        images = [img.to(device) for img in images]
        outputs = model(images)
        for i, (img_id, scale) in enumerate(zip(img_ids, scales)):
            pred = outputs[i]
            boxes = pred['boxes'].cpu().numpy()
            scores = pred['scores'].cpu().numpy()
            labels = pred['labels'].cpu().numpy()
            keep = scores >= CONF_THRESH
            boxes = boxes[keep]
            labels = labels[keep]
            scores = scores[keep]

            if len(boxes) > MAX_PREDS:
                top_idx = np.argsort(scores)[-MAX_PREDS:][::-1]
                boxes = boxes[top_idx]
                labels = labels[top_idx]
                scores = scores[top_idx]

            if len(boxes) == 0:
                submission_dict[img_id] = ''
                continue

            # Scale back to original coordinates
            boxes = boxes / scale
            centers_x = (boxes[:, 0] + boxes[:, 2]) / 2
            centers_y = (boxes[:, 1] + boxes[:, 3]) / 2

            pred_strings = []
            for lbl, x, y in zip(labels, centers_x, centers_y):
                char = id_to_char.get(lbl)
                if char is None:
                    continue
                x_int = int(round(x))
                y_int = int(round(y))
                pred_strings.append(f"{char} {x_int} {y_int}")
            submission_dict[img_id] = ' '.join(pred_strings)

# Write submission file
submission_df = pd.DataFrame({
    'image_id': list(submission_dict.keys()),
    'labels': list(submission_dict.values())
})
submission_df.to_csv(os.path.join(SUBMISSION_DIR, 'submission.csv'), index=False)
print("Submission file saved to submission/submission.csv")
print("Done.")