import os
import sys
import glob
import random
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Configuration
BATCH_SIZE = 64
IMAGE_SIZE = 224
RESIZE = 256
EPOCHS = 5
LR = 1e-4
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Load metadata and create class mapping
train_df = pd.read_csv("input/train.csv")
unique_hotel_ids = train_df["hotel_id"].unique()
hotel_to_idx = {hotel: idx for idx, hotel in enumerate(unique_hotel_ids)}
idx_to_hotel = {idx: hotel for hotel, idx in hotel_to_idx.items()}
num_classes = len(unique_hotel_ids)
print(f"Number of classes (hotels): {num_classes}")

# Build full image paths
train_df["path"] = train_df.apply(lambda row: f"input/train_images/{row['chain']}/{row['image']}", axis=1)
pairs = list(zip(train_df["path"], train_df["hotel_id"].map(hotel_to_idx)))

# Train/validation split (90/10)
train_pairs, val_pairs = train_test_split(pairs, test_size=0.1, random_state=SEED, shuffle=True)
print(f"Train samples: {len(train_pairs)}, Validation samples: {len(val_pairs)}")

# Image transformations
train_transform = transforms.Compose([
    transforms.Resize((RESIZE, RESIZE)),
    transforms.RandomCrop(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((RESIZE, RESIZE)),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Dataset classes
class HotelDataset(Dataset):
    def __init__(self, pairs, transform=None):
        self.pairs = pairs
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        path, label = self.pairs[idx]
        try:
            img = Image.open(path).convert('RGB')
        except Exception:
            img = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), (0, 0, 0))
        if self.transform:
            img = self.transform(img)
        return img, label

class TestDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            img = Image.open(path).convert('RGB')
        except Exception:
            img = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), (0, 0, 0))
        if self.transform:
            img = self.transform(img)
        return img, path

# Data loaders
train_dataset = HotelDataset(train_pairs, train_transform)
val_dataset = HotelDataset(val_pairs, val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)

# Model
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1)

# MAP@5 metric
def map5(outputs, targets):
    """Compute MAP@5 for a batch."""
    _, top5 = outputs.topk(5, dim=1)  # (batch, 5)
    ap_scores = []
    for i in range(targets.size(0)):
        true = targets[i].item()
        preds = top5[i].tolist()
        try:
            rank = preds.index(true) + 1
            ap = 1.0 / rank
        except ValueError:
            ap = 0.0
        ap_scores.append(ap)
    return np.mean(ap_scores)

# Training loop
best_map = 0.0
history = []

for epoch in range(1, EPOCHS+1):
    print(f"\nEpoch {epoch}/{EPOCHS}")
    model.train()
    running_loss = 0.0
    progress = tqdm(train_loader, desc="Training", leave=False)
    for images, labels in progress:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        progress.set_postfix(loss=loss.item())
    train_loss = running_loss / len(train_loader.dataset)

    # Validation
    model.eval()
    val_loss = 0.0
    map_sum = 0.0
    count = 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validation", leave=False):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            map_sum += map5(outputs, labels) * images.size(0)
            count += images.size(0)
    val_loss /= len(val_loader.dataset)
    val_map = map_sum / len(val_loader.dataset) if count > 0 else 0.0
    scheduler.step(val_map)

    history.append((train_loss, val_loss, val_map))
    print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val MAP@5: {val_map:.4f}")

    # Save best model
    if val_map > best_map:
        best_map = val_map
        os.makedirs("working", exist_ok=True)
        torch.save(model.state_dict(), "working/best_model.pth")
        print(f"Best model saved with MAP@5: {best_map:.4f}")

print(f"\nBest Validation MAP@5: {best_map:.4f}")

# Load best model for inference
model.load_state_dict(torch.load("working/best_model.pth", map_location=DEVICE))
model.eval()

# Process test images
test_images = sorted(glob.glob("input/test_images/*.jpg"))
print(f"Found {len(test_images)} test images.")
test_dataset = TestDataset(test_images, val_transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=True)

predictions = []
with torch.no_grad():
    for images, paths in tqdm(test_loader, desc="Predicting"):
        images = images.to(DEVICE)
        outputs = model(images)
        _, top5 = outputs.topk(5, dim=1)
        for i in range(images.size(0)):
            idxs = top5[i].cpu().numpy()
            hotels = [idx_to_hotel[idx] for idx in idxs]
            filename = os.path.basename(paths[i])
            predictions.append((filename, " ".join(map(str, hotels))))

# Save submission
sub_df = pd.DataFrame(predictions, columns=["image", "hotel_id"])
os.makedirs("submission", exist_ok=True)
sub_df.to_csv("submission/submission.csv", index=False)
print("Submission saved to submission/submission.csv")

print("Done.")