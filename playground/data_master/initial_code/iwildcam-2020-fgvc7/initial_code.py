import os
import json
import random
import warnings
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from sklearn.model_selection import GroupKFold

warnings.filterwarnings('ignore')

# -------------------- Configuration --------------------
RANDOM_SEED = 42
BATCH_SIZE = 256
NUM_WORKERS = 8
IMG_SIZE = 224
LR = 1e-4
WEIGHT_DECAY = 0.05
EPOCHS = 5
NUM_SPLITS = 5
VAL_FOLD = 0
EMPTY_THRESH = 0.05   # confidence threshold for forcing empty class

# Paths
INPUT_DIR = './input'
TRAIN_ANN_FILE = os.path.join(INPUT_DIR, 'iwildcam2020_train_annotations.json')
MEGADETECTOR_FILE = os.path.join(INPUT_DIR, 'iwildcam2020_megadetector_results.json')
TEST_INFO_FILE = os.path.join(INPUT_DIR, 'iwildcam2020_test_information.json')
TRAIN_IMG_DIR = os.path.join(INPUT_DIR, 'train')
TEST_IMG_DIR = os.path.join(INPUT_DIR, 'test')
WORKING_DIR = './working'
SUBMISSION_DIR = './submission'
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# -------------------- Data Loading --------------------
print('Loading metadata...')
with open(TRAIN_ANN_FILE) as f:
    train_ann = json.load(f)
with open(MEGADETECTOR_FILE) as f:
    md_data = json.load(f)
with open(TEST_INFO_FILE) as f:
    test_info = json.load(f)

# Build training DataFrame
images_df = pd.DataFrame(train_ann['images'])
annotations_df = pd.DataFrame(train_ann['annotations'])

# Merge with suffixes to avoid column name conflicts
train_df = pd.merge(images_df, annotations_df, left_on='id', right_on='image_id', suffixes=('_img', '_ann'))

# Drop the duplicate image_id column from annotations (right key)
train_df.drop(columns=['image_id'], inplace=True)

# Rename id_img to image_id
train_df.rename(columns={'id_img': 'image_id'}, inplace=True)

# Drop annotation id (id_ann)
train_df.drop(columns=['id_ann'], inplace=True)

# MegaDetector confidence mapping
md_conf = {img['id']: img['max_detection_conf'] for img in md_data['images']}
train_df['conf'] = train_df['image_id'].map(md_conf).fillna(0.0)

# Number of classes (including 0)
categories = train_ann['categories']
num_classes = max(cat['id'] for cat in categories) + 1
print(f'Number of classes: {num_classes}')

# -------------------- Train/Validation Split (GroupKFold) --------------------
gkf = GroupKFold(n_splits=NUM_SPLITS)
fold_assign = list(gkf.split(train_df, groups=train_df['location']))
train_idx, val_idx = fold_assign[VAL_FOLD]
train_subset = train_df.iloc[train_idx].reset_index(drop=True)
val_subset = train_df.iloc[val_idx].reset_index(drop=True)
print(f'Train size: {len(train_subset)}, Val size: {len(val_subset)}')

# Class weights for weighted sampler (to handle imbalance)
class_counts = train_subset['category_id'].value_counts()
weight_dict = {cls: 1. / count for cls, count in class_counts.items()}
sample_weights = torch.tensor([weight_dict[cls] for cls in train_subset['category_id']], dtype=torch.float)
sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_subset), replacement=True)

# -------------------- Transforms --------------------
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# -------------------- Dataset --------------------
class iWildCamDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, is_test=False):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row['image_id'] if not self.is_test else row['id']
        img_path = os.path.join(self.img_dir, f'{img_id}.jpg')
        try:
            img = Image.open(img_path).convert('RGB')
        except:
            # fallback to a black image if file missing/corrupt
            img = Image.new('RGB', (IMG_SIZE, IMG_SIZE), color=(0,0,0))

        if self.transform:
            img = self.transform(img)

        conf = torch.tensor(row['conf'], dtype=torch.float)

        if self.is_test:
            return img, conf, img_id
        else:
            label = torch.tensor(row['category_id'], dtype=torch.long)
            return img, conf, label

# Create datasets & dataloaders
train_dataset = iWildCamDataset(train_subset, TRAIN_IMG_DIR, transform=train_transform)
val_dataset = iWildCamDataset(val_subset, TRAIN_IMG_DIR, transform=val_transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

# -------------------- Model --------------------
class ConvNextWithConfidence(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        backbone = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
        self.features = backbone.features
        # Image features dimension for tiny variant is 768
        self.img_feat_dim = 768
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        # Project confidence (1 -> 32) and concatenate
        self.conf_proj = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU()
        )
        self.classifier = nn.Linear(self.img_feat_dim + 32, num_classes)

    def forward(self, x, conf):
        # x: image tensor
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)                 # (batch, 768)
        conf_feat = self.conf_proj(conf.unsqueeze(1))  # (batch, 32)
        combined = torch.cat((x, conf_feat), dim=1)
        out = self.classifier(combined)
        return out

model = ConvNextWithConfidence(num_classes=num_classes).to(device)
if hasattr(torch, 'compile'):
    model = torch.compile(model)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
scaler = torch.cuda.amp.GradScaler()

# -------------------- Training --------------------
best_val_acc = 0.0
best_model_path = os.path.join(WORKING_DIR, 'best_model.pth')

print('Starting training...')
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS} Train')
    for images, confs, labels in pbar:
        images = images.to(device, non_blocking=True)
        confs = confs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(images, confs)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        pbar.set_postfix(loss=loss.item())

    train_loss = running_loss / len(train_loader.dataset)

    # Validation
    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0
    with torch.no_grad():
        for images, confs, labels in tqdm(val_loader, desc=f'Epoch {epoch+1}/{EPOCHS} Val'):
            images = images.to(device, non_blocking=True)
            confs = confs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.cuda.amp.autocast():
                outputs = model(images, confs)
                loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_loss /= len(val_loader.dataset)
    val_acc = correct / total
    print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_model_path)
        print(f'Best model saved with val acc: {best_val_acc:.4f}')

    scheduler.step()

print(f'Training finished. Best Val Acc: {best_val_acc:.4f}')

# -------------------- Test Inference --------------------
# Prepare test DataFrame
test_images_df = pd.DataFrame(test_info['images'])
test_images_df['conf'] = test_images_df['id'].map(md_conf).fillna(0.0)

test_dataset = iWildCamDataset(test_images_df, TEST_IMG_DIR, transform=val_transform, is_test=True)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

# Load best model
model.load_state_dict(torch.load(best_model_path))
model.eval()

all_preds = []
all_ids = []
with torch.no_grad():
    for images, confs, img_ids in tqdm(test_loader, desc='Test Inference'):
        images = images.to(device, non_blocking=True)
        confs = confs.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            outputs = model(images, confs)
        _, preds = outputs.max(1)
        # Override with class 0 if confidence is below threshold
        preds = torch.where(confs < EMPTY_THRESH, torch.zeros_like(preds), preds)
        all_preds.append(preds.cpu())
        all_ids.extend(list(img_ids))

all_preds = torch.cat(all_preds).numpy()

# -------------------- Save Submission --------------------
submission_df = pd.DataFrame({'Id': all_ids, 'Category': all_preds})
submission_path = os.path.join(SUBMISSION_DIR, 'submission.csv')
submission_df.to_csv(submission_path, index=False)
print(f'Submission saved to {submission_path}')

# Print final validation metric (required)
print(f'Best validation accuracy: {best_val_acc:.4f}')