import os
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from PIL import Image

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42)

# Constants
BATCH_SIZE = 128
IMG_SIZE = 224
NUM_EPOCHS = 10
LR = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 8
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE_DIR = Path("./input/nybg2020")

# Create necessary directories
os.makedirs("./submission", exist_ok=True)
os.makedirs("./working", exist_ok=True)

# ------------------------------------------------------------
# Load training metadata and build DataFrame
# ------------------------------------------------------------
print("Loading training metadata...")
with open(BASE_DIR / "train/metadata.json") as f:
    train_meta = json.load(f)

# Mapping image_id -> file_name
img_id_to_file = {img['id']: img['file_name'] for img in train_meta['images']}
annots = train_meta['annotations']

rows = []
for ann in annots:
    img_id = ann['image_id']
    cat_id = ann['category_id']
    if img_id in img_id_to_file:
        img_path = BASE_DIR / "train" / img_id_to_file[img_id]
        rows.append((img_id, cat_id, str(img_path)))
    else:
        print(f"Warning: image id {img_id} not found in images metadata")
df = pd.DataFrame(rows, columns=['image_id', 'category_id', 'image_path'])

print(f"Total training samples: {len(df)}")
class_counts = df['category_id'].value_counts()
print(f"Number of classes: {len(class_counts)}")
print(f"Classes with 1 sample: {len(class_counts[class_counts == 1])}")

# ------------------------------------------------------------
# Stratified train / validation split
# ------------------------------------------------------------
single_classes = class_counts[class_counts == 1].index.tolist()
multi_classes = class_counts[class_counts >= 2].index.tolist()

df_single = df[df['category_id'].isin(single_classes)]
df_multi = df[df['category_id'].isin(multi_classes)]

train_multi, val_multi = train_test_split(
    df_multi, test_size=0.1, stratify=df_multi['category_id'], random_state=42
)

train_df = pd.concat([train_multi, df_single])
val_df = val_multi

print(f"Train size: {len(train_df)}, Val size: {len(val_df)}")

# ------------------------------------------------------------
# Map category_id to contiguous indices
# ------------------------------------------------------------
all_cats = sorted(df['category_id'].unique())
cat_to_idx = {cat: idx for idx, cat in enumerate(all_cats)}
idx_to_cat = {idx: cat for cat, idx in cat_to_idx.items()}
num_classes = len(all_cats)
print(f"Number of classes: {num_classes}")

train_df['category_idx'] = train_df['category_id'].map(cat_to_idx)
val_df['category_idx'] = val_df['category_id'].map(cat_to_idx)

# ------------------------------------------------------------
# Class weights for weighted random sampling
# ------------------------------------------------------------
class_counts_train = train_df['category_idx'].value_counts()
weights = 1.0 / class_counts_train
sample_weights = train_df['category_idx'].map(weights).values
sample_weights = torch.from_numpy(sample_weights).float()

# ------------------------------------------------------------
# Dataset and DataLoader definitions
# ------------------------------------------------------------
class HerbariumDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['image_path']
        label = row['category_idx']
        try:
            img = Image.open(img_path).convert('RGB')
        except:
            img = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (0, 0, 0))
        if self.transform:
            img = self.transform(img)
        return img, label

class TestDataset(Dataset):
    def __init__(self, image_list, transform=None):
        self.image_list = image_list  # list of (img_id, img_path)
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_id, img_path = self.image_list[idx]
        try:
            img = Image.open(img_path).convert('RGB')
        except:
            img = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (0, 0, 0))
        if self.transform:
            img = self.transform(img)
        return img, img_id

# Transforms
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_ds = HerbariumDataset(train_df, transform=train_transform)
val_ds = HerbariumDataset(val_df, transform=val_transform)

# Weighted sampler for training
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(train_df),
    replacement=True
)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

# ------------------------------------------------------------
# Model, loss, optimizer, scheduler
# ------------------------------------------------------------
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(DEVICE)

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

criterion = FocalLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

# ------------------------------------------------------------
# Training loop with validation macro F1
# ------------------------------------------------------------
best_f1 = 0.0
scaler = torch.amp.GradScaler('cuda')  # updated API

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    train_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):  # updated API
            outputs = model(images)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item() * images.size(0)
    train_loss /= len(train_loader.dataset)
    scheduler.step()

    # Validation
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    val_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    print(f"Epoch {epoch}/{NUM_EPOCHS} - Train loss: {train_loss:.4f}, Val macro F1: {val_f1:.4f}")
    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(model.state_dict(), "./working/best_model.pth")
        print(f"Saved best model with val_f1 {val_f1:.4f}")

print(f"\nBest validation macro F1: {best_f1:.4f}")

# ------------------------------------------------------------
# Test inference and submission
# ------------------------------------------------------------
print("Loading test metadata...")
with open(BASE_DIR / "test/metadata.json") as f:
    test_meta = json.load(f)

test_images = []
for img in test_meta['images']:
    img_id = img['id']
    file_name = img['file_name']
    img_path = BASE_DIR / "test" / file_name
    test_images.append((img_id, str(img_path)))

test_ds = TestDataset(test_images, transform=val_transform)
test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

model.load_state_dict(torch.load("./working/best_model.pth"))
model.eval()
predictions = []
with torch.no_grad():
    for images, img_ids in test_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        for img_id, pred_idx in zip(img_ids, preds):
            pred_cat = idx_to_cat[pred_idx]
            # Convert to int safely (handles tensor and string)
            predictions.append((int(img_id), pred_cat))

sub_df = pd.DataFrame(predictions, columns=['Id', 'Predicted'])
sub_df.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved to ./submission/submission.csv with {len(sub_df)} rows.")
print("Done.")