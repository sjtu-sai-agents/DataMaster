import os
import random
import sys
import warnings

import numpy as np
import pandas as pd
import PIL
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from PIL import Image, ImageFile
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm.auto import tqdm

# ----------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------
SEED = 42

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

set_seed()

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
BASE_DIR = "./input"
TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
TEST_CSV = os.path.join(BASE_DIR, "test.csv")
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "train_images")
TEST_IMG_DIR = os.path.join(BASE_DIR, "test_images")
SUBMISSION_PATH = "./submission/submission.csv"

IMG_SIZE = 224
BATCH_SIZE = 64
NUM_EPOCHS = 5
LR = 3e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.05
GRAD_CLIP = 5.0
VALID_SIZE = 0.1
NUM_WORKERS = min(8, max(2, (os.cpu_count() or 2) // 2))
os.makedirs("./submission", exist_ok=True)

ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.multiprocessing.set_sharing_strategy("file_system")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----------------------------------------------------------------------
# Data preparation
# ----------------------------------------------------------------------
def filter_existing(df, img_dir, fname_col="file_name"):
    """Keep only rows whose image file exists on disk."""
    total = len(df)
    paths = df[fname_col].apply(lambda x: os.path.join(img_dir, x))
    mask = paths.map(os.path.exists)
    missing = total - mask.sum()
    if missing:
        print(f"Dropping {missing} rows with missing images.")
    return df[mask].copy()

# Load CSVs
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

# Filter for existing files
train_df = filter_existing(train_df, TRAIN_IMG_DIR)
test_df = filter_existing(test_df, TEST_IMG_DIR)

# Number of classes (0..22)
num_classes = int(train_df["category_id"].max()) + 1
print(f"Number of classes: {num_classes}")

# ----------------------------------------------------------------------
# Train / Validation split (stratified)
# ----------------------------------------------------------------------
X = train_df.index.values
y = train_df["category_id"].values

try:
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=VALID_SIZE, random_state=SEED)
    train_idx, val_idx = next(splitter.split(X, y))
except ValueError:   # stratification impossible (single class?)
    print("Stratified split failed, falling back to random split")
    splitter = ShuffleSplit(n_splits=1, test_size=VALID_SIZE, random_state=SEED)
    train_idx, val_idx = next(splitter.split(X))

train_split_df = train_df.iloc[train_idx].reset_index(drop=True)
val_split_df = train_df.iloc[val_idx].reset_index(drop=True)

print(f"Training samples: {len(train_split_df)}")
print(f"Validation samples: {len(val_split_df)}")

# ----------------------------------------------------------------------
# Transforms
# ----------------------------------------------------------------------
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_tfms = T.Compose([
    T.Resize(256),
    T.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0), ratio=(0.8, 1.25)),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
    T.ToTensor(),
    T.Normalize(mean, std)
])

val_tfms = T.Compose([
    T.Resize(256),
    T.CenterCrop(IMG_SIZE),
    T.ToTensor(),
    T.Normalize(mean, std)
])

# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------
class CameraTrapDataset(Dataset):
    def __init__(self, df, img_dir, tfms, label_col=None, fname_col="file_name", id_col="id"):
        self.df = df
        self.img_dir = img_dir
        self.tfms = tfms
        self.label_col = label_col
        self.fname_col = fname_col
        self.id_col = id_col

    def _safe_open(self, path):
        try:
            img = Image.open(path).convert('RGB')
            return img
        except Exception as e:
            warnings.warn(f"Error loading {path}: {e}. Returning blank image.")
            return Image.new('RGB', (256, 256), (0,0,0))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row[self.fname_col])
        image = self._safe_open(img_path)
        image = self.tfms(image)

        if self.label_col is not None:
            label = int(row[self.label_col])
            return image, label
        else:
            sample_id = row[self.id_col]
            return image, sample_id

# ----------------------------------------------------------------------
# DataLoaders
# ----------------------------------------------------------------------
train_dataset = CameraTrapDataset(train_split_df, TRAIN_IMG_DIR, train_tfms, label_col="category_id")
val_dataset = CameraTrapDataset(val_split_df, TRAIN_IMG_DIR, val_tfms, label_col="category_id")
test_dataset = CameraTrapDataset(test_df, TEST_IMG_DIR, val_tfms, label_col=None)

# Weighted sampler for class imbalance
class_counts = train_split_df["category_id"].value_counts().sort_index()
class_weights_dict = {cat: 1.0 / count for cat, count in class_counts.items()}
sample_weights = [class_weights_dict[c] for c in train_split_df["category_id"]]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_dataset), replacement=True)

# Create loaders with fallback to 0 workers if any error occurs
try:
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
except Exception as e:
    print(f"Dataloader error: {e}, falling back to 0 workers")
    NUM_WORKERS = 0
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------
if hasattr(torchvision.models, "ResNet50_Weights"):
    weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
else:
    weights = None

model = torchvision.models.resnet50(weights=weights)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)
model = model.to(device)

# Loss, Optimizer, Scheduler
criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

# Mixed precision scaler (only for CUDA)
if device.type == "cuda":
    scaler = torch.cuda.amp.GradScaler()
else:
    scaler = None

# ----------------------------------------------------------------------
# Training / Validation
# ----------------------------------------------------------------------
best_f1 = 0.0
best_state = None

for epoch in range(1, NUM_EPOCHS + 1):
    # ------------------------- Train -------------------------
    model.train()
    train_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [Train]", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

        train_loss += loss.item() * images.size(0)
        pbar.set_postfix(loss=loss.item())

    train_loss /= len(train_loader.dataset)
    scheduler.step()

    # ------------------------- Validation -------------------------
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [Val]", leave=False)
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=(scaler is not None)):
                logits = model(images)
                flipped = torch.flip(images, dims=[-1])
                logits_flip = model(flipped)
                avg_logits = (logits + logits_flip) / 2.0

            preds = avg_logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    val_f1 = f1_score(all_labels, all_preds, average="macro")
    print(f"Epoch {epoch}: Train loss = {train_loss:.4f}, Val F1 = {val_f1:.5f}")

    if val_f1 > best_f1:
        best_f1 = val_f1
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        torch.save(best_state, "best_model.pth")

# ----------------------------------------------------------------------
# Load best model and predict on test set
# ----------------------------------------------------------------------
if best_state is not None:
    model.load_state_dict(best_state)
model.eval()

test_ids = []
test_preds = []

with torch.no_grad():
    pbar = tqdm(test_loader, desc="Test Inference")
    for images, ids in pbar:
        images = images.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            logits = model(images)
            flipped = torch.flip(images, dims=[-1])
            logits_flip = model(flipped)
            avg_logits = (logits + logits_flip) / 2.0

        preds = avg_logits.argmax(dim=1).cpu().numpy()
        test_preds.extend(preds)
        test_ids.extend(ids)

# Check alignment
assert len(test_ids) == len(test_df), f"Predicted {len(test_ids)} vs test {len(test_df)}"

# ----------------------------------------------------------------------
# Create submission
# ----------------------------------------------------------------------
submission_df = pd.DataFrame({
    "Id": test_ids,
    "Category": test_preds
})
submission_df.to_csv(SUBMISSION_PATH, index=False)
print(f"Submission saved to {SUBMISSION_PATH}")

# ----------------------------------------------------------------------
# Print final evaluation metric
# ----------------------------------------------------------------------
print(f"\nBest validation macro F1: {best_f1:.5f}")