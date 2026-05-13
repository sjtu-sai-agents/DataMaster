import os
import json
import random
from collections import Counter
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import timm
from tqdm import tqdm

# Constants
IMG_SIZE = 224
BATCH_SIZE = 128
EPOCHS = 5
LR = 5e-4
WEIGHT_DECAY = 0.05
LABEL_SMOOTHING = 0.1
VAL_SIZE = 0.05
SEED = 42

# Enable loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Set random seeds for reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True

# Create necessary directories
os.makedirs("./working", exist_ok=True)
os.makedirs("./submission", exist_ok=True)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------------------------------------------
# Data path helpers
def get_train_samples(metadata_path):
    with open(metadata_path) as f:
        meta = json.load(f)
    annotations = meta['annotations']
    samples = []
    for ann in annotations:
        img_id = ann['image_id']
        cat_id = ann['category_id']
        cat_str = f"{cat_id:05d}"
        sub1, sub2 = cat_str[:3], cat_str[3:]
        path = os.path.join("./input/train/images", sub1, sub2, f"{img_id}.jpg")
        samples.append((path, cat_id))
    return samples

def get_test_samples(metadata_path):
    with open(metadata_path) as f:
        meta = json.load(f)
    images = meta['images']
    samples = []
    for img in images:
        img_id = int(img['id'])                 # Convert to int (FIXED)
        sub = f"{img_id // 1000:03d}"
        path = os.path.join("./input/test/images", sub, f"{img_id}.jpg")
        samples.append((path, img_id))
    return samples

# ----------------------------------------------------------------------
# Dataset class
class HerbariumDataset(Dataset):
    def __init__(self, samples, transform=None, is_test=False):
        self.samples = samples        # list of (path, target)
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        try:
            img = Image.open(path).convert('RGB')
        except:
            img = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (0,0,0))
        if self.transform:
            img = self.transform(img)
        # For test set, target is the image id; otherwise it's the encoded label
        return img, target

# ----------------------------------------------------------------------
# Load training data and prepare splits
print("Loading training metadata...")
train_samples = get_train_samples("./input/train/metadata.json")
paths = [p for p,_ in train_samples]
orig_cats = [c for _,c in train_samples]

# Label encoding
le = LabelEncoder()
encoded_cats = le.fit_transform(orig_cats)
num_classes = len(le.classes_)
print(f"Number of classes: {num_classes}")

# Group indices by category
cat_to_indices = {}
for idx, cat in enumerate(orig_cats):
    cat_to_indices.setdefault(cat, []).append(idx)

multi_indices = []
single_indices = []
for cat, idxs in cat_to_indices.items():
    if len(idxs) >= 2:
        multi_indices.extend(idxs)
    else:
        single_indices.extend(idxs)

# Separate multi-sample data
multi_paths = [paths[i] for i in multi_indices]
multi_encoded = [encoded_cats[i] for i in multi_indices]

# Stratified split on multi-sample data
train_paths_multi, val_paths, train_labels_multi, val_labels = train_test_split(
    multi_paths, multi_encoded, test_size=VAL_SIZE, stratify=multi_encoded, random_state=SEED
)

# Single-sample data -> training only
train_paths_single = [paths[i] for i in single_indices]
train_labels_single = [encoded_cats[i] for i in single_indices]

# Combine training data
train_paths = train_paths_multi + train_paths_single
train_labels = train_labels_multi + train_labels_single

print(f"Training samples: {len(train_paths)}")
print(f"Validation samples: {len(val_paths)}")

# ----------------------------------------------------------------------
# Transforms
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize(int(IMG_SIZE * 1.14)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Datasets & DataLoaders
train_dataset = HerbariumDataset(list(zip(train_paths, train_labels)), transform=train_transform, is_test=False)
val_dataset = HerbariumDataset(list(zip(val_paths, val_labels)), transform=val_transform, is_test=False)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=8, pin_memory=True, persistent_workers=True
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=8, pin_memory=True, persistent_workers=True
)

# ----------------------------------------------------------------------
# Model
model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

steps_per_epoch = len(train_loader)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=LR, epochs=EPOCHS, steps_per_epoch=steps_per_epoch,
    pct_start=0.1, anneal_strategy='cos'
)

scaler = torch.cuda.amp.GradScaler()

# ----------------------------------------------------------------------
# Validation function
def validate(model, loader, le):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating"):
            images = images.to(device)
            labels = labels.to(device)
            with torch.cuda.amp.autocast():
                outputs = model(images)
            _, predicted = outputs.max(1)
            preds.extend(predicted.cpu().numpy())
            trues.extend(labels.cpu().numpy())
    preds_orig = le.inverse_transform(preds)
    trues_orig = le.inverse_transform(trues)
    return f1_score(trues_orig, preds_orig, average='macro')

# ----------------------------------------------------------------------
# Training loop
best_f1 = 0.0
for epoch in range(EPOCHS):
    model.train()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        pbar.set_postfix(loss=loss.item())
    val_f1 = validate(model, val_loader, le)
    print(f"Epoch {epoch+1} - Validation macro F1: {val_f1:.4f}")
    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(model.state_dict(), "./working/best_model.pth")
        print("  -> New best model saved.")

# ----------------------------------------------------------------------
# Load best model for test inference
model.load_state_dict(torch.load("./working/best_model.pth", map_location=device))
model.eval()

print("Loading test metadata...")
test_samples = get_test_samples("./input/test/metadata.json")
test_dataset = HerbariumDataset(test_samples, transform=val_transform, is_test=True)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=8, pin_memory=True, persistent_workers=True
)

all_ids = []
all_preds = []
with torch.no_grad():
    for images, ids in tqdm(test_loader, desc="Inferring"):
        images = images.to(device)
        with torch.cuda.amp.autocast():
            outputs = model(images)
        _, preds = outputs.max(1)
        all_ids.extend(ids.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

preds_orig = le.inverse_transform(all_preds)

# ----------------------------------------------------------------------
# Save submission
sub_df = pd.DataFrame({"Id": all_ids, "Predicted": preds_orig})
sub_df.to_csv("./submission/submission.csv", index=False)
print(f"Saved submission with {len(sub_df)} rows.")

print(f"Best validation macro F1: {best_f1:.4f}")