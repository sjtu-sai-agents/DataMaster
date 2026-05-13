import os
import json
import random
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.cuda.amp import autocast, GradScaler

import timm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score

# Configuration
CONFIG = {
    'seed': 42,
    'img_size': 320,
    'batch_size': 64,
    'num_workers': 8,
    'lr': 1e-3,
    'epochs': 5,
    'num_classes': 15501,
    'label_smoothing': 0.1,
    'val_size': 0.1,
    'use_amp': True,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'train_meta_path': 'input/train_metadata.json',
    'test_meta_path': 'input/test_metadata.json',
    'train_img_dir': 'input/train_images',
    'test_img_dir': 'input/test_images',
    'submission_path': 'submission/submission.csv',
}

# Set seeds
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

set_seed(CONFIG['seed'])

# Load and prepare data
print("Loading training metadata...")
with open(CONFIG['train_meta_path'], 'r') as f:
    train_meta = json.load(f)

images_df = pd.DataFrame(train_meta['images'])
annotations_df = pd.DataFrame(train_meta['annotations'])
train_df = pd.merge(images_df, annotations_df, on='image_id')

# Encode labels
le = LabelEncoder()
train_df['label'] = le.fit_transform(train_df['category_id'])

# Stratified split
try:
    train_df, val_df = train_test_split(
        train_df, test_size=CONFIG['val_size'],
        stratify=train_df['label'], random_state=CONFIG['seed']
    )
except ValueError:
    print("Stratification failed, falling back to random split.")
    train_df, val_df = train_test_split(
        train_df, test_size=CONFIG['val_size'],
        random_state=CONFIG['seed']
    )

print(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")

# Load test metadata
print("Loading test metadata...")
with open(CONFIG['test_meta_path'], 'r') as f:
    test_meta = json.load(f)

if isinstance(test_meta, list):
    test_images = test_meta
else:
    test_images = test_meta['images']

test_df = pd.DataFrame(test_images)
test_df['image_id'] = test_df['image_id'].astype(int)
print(f"Test samples: {len(test_df)}")

# Dataset
class HerbariumDataset(Dataset):
    def __init__(self, df, img_dir, transforms=None, is_test=False):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transforms = transforms
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row['image_id']
        file_name = row['file_name']

        # Construct full path
        img_path = os.path.join(self.img_dir, file_name)

        try:
            img = Image.open(img_path).convert('RGB')
        except:
            # corrupted image, return blank
            img = Image.new('RGB', (CONFIG['img_size'], CONFIG['img_size']), (0,0,0))

        if self.transforms:
            img = self.transforms(img)

        if self.is_test:
            return img, img_id
        else:
            label = row['label']
            return img, label

# Transforms
train_transform = transforms.Compose([
    transforms.Resize((CONFIG['img_size'], CONFIG['img_size'])),
    transforms.RandAugment(num_ops=2, magnitude=9),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((CONFIG['img_size'], CONFIG['img_size'])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

test_transform = val_transform

# DataLoaders
train_dataset = HerbariumDataset(train_df, CONFIG['train_img_dir'], train_transform, is_test=False)
val_dataset = HerbariumDataset(val_df, CONFIG['train_img_dir'], val_transform, is_test=False)
test_dataset = HerbariumDataset(test_df, CONFIG['test_img_dir'], test_transform, is_test=True)

train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'],
                          shuffle=True, num_workers=CONFIG['num_workers'],
                          pin_memory=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'],
                        shuffle=False, num_workers=CONFIG['num_workers'],
                        pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'],
                         shuffle=False, num_workers=CONFIG['num_workers'],
                         pin_memory=True)

# Model
model = timm.create_model('tf_efficientnet_b3_ns', pretrained=True,
                          num_classes=CONFIG['num_classes'])
model = model.to(CONFIG['device'])

# Loss, optimizer, scheduler
criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG['label_smoothing'])
optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=CONFIG['epochs'], eta_min=1e-6
)

scaler = GradScaler(enabled=CONFIG['use_amp'])

# Training
best_f1 = 0.0
for epoch in range(CONFIG['epochs']):
    model.train()
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(CONFIG['device'], non_blocking=True)
        labels = labels.to(CONFIG['device'], non_blocking=True)

        optimizer.zero_grad()

        with autocast(enabled=CONFIG['use_amp']):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        if (i+1) % 500 == 0:
            avg_loss = running_loss / (i+1)
            print(f"Epoch {epoch+1}/{CONFIG['epochs']}, Step {i+1}, Loss: {avg_loss:.4f}")

    scheduler.step()

    # Validation
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(CONFIG['device'], non_blocking=True)
            with autocast(enabled=CONFIG['use_amp']):
                outputs = model(images)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    val_f1 = f1_score(all_labels, all_preds, average='macro')
    print(f"Epoch {epoch+1} Validation Macro F1: {val_f1:.6f}")

    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(model.state_dict(), 'best_model.pth')
        print("Saved best model.")

print(f"Best validation Macro F1: {best_f1:.6f}")

# Load best model for test inference
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# Test prediction
all_ids = []
all_preds = []
with torch.no_grad():
    for images, ids in test_loader:
        images = images.to(CONFIG['device'], non_blocking=True)
        with autocast(enabled=CONFIG['use_amp']):
            outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_ids.extend(ids.numpy())

# Decode to original category_id
decoded_preds = le.inverse_transform(all_preds)

# Create submission
submission_df = pd.DataFrame({'Id': all_ids, 'Predicted': decoded_preds})
submission_df = submission_df.sort_values('Id')
os.makedirs(os.path.dirname(CONFIG['submission_path']), exist_ok=True)
submission_df.to_csv(CONFIG['submission_path'], index=False)
print("Submission saved to", CONFIG['submission_path'])

# Print evaluation metric (best validation F1)
print(f"Evaluation metric (macro F1 on validation set): {best_f1:.6f}")