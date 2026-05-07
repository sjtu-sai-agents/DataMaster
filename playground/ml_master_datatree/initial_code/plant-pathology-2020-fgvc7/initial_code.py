import os
import random
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Set reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42)

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15
LR = 0.001
T_MAX = 10  # for cosine annealing
NUM_WORKERS = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
train_df = pd.read_csv('./input/train.csv')
test_df = pd.read_csv('./input/test.csv')

# Train/validation split (80/20)
train_df, val_df = train_test_split(
    train_df, test_size=0.2, random_state=42, shuffle=True
)

# Transforms
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

test_transform = val_transform

# Dataset class
class PlantDataset(Dataset):
    def __init__(self, df, transform=None, is_test=False):
        self.df = df
        self.transform = transform
        self.is_test = is_test
        self.image_dir = "./input/images"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row['image_id']
        img_path = os.path.join(self.image_dir, f"{image_id}.jpg")
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        if self.is_test:
            return image, image_id
        else:
            labels = row[['healthy', 'multiple_diseases', 'rust', 'scab']].values.astype(np.float32)
            return image, torch.from_numpy(labels)

# DataLoaders
train_ds = PlantDataset(train_df, transform=train_transform, is_test=False)
val_ds = PlantDataset(val_df, transform=val_transform, is_test=False)
test_ds = PlantDataset(test_df, transform=test_transform, is_test=True)

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=True
)
val_loader = DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True
)
test_loader = DataLoader(
    test_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True
)

# Model
model = models.efficientnet_b0(pretrained=True)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 4)
model = model.to(device)

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_MAX)

# Training
best_auc = 0.0
best_state = None

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)
    train_loss /= len(train_loader.dataset)
    scheduler.step()

    # Validation
    model.eval()
    val_loss = 0.0
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            probs = torch.sigmoid(outputs)
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    val_loss /= len(val_loader.dataset)
    all_labels = np.concatenate(all_labels, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)

    auc_scores = []
    for i in range(4):
        try:
            auc = roc_auc_score(all_labels[:, i], all_probs[:, i])
        except ValueError:
            auc = 0.5
        auc_scores.append(auc)
    mean_auc = np.mean(auc_scores)

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {mean_auc:.4f}")

    if mean_auc > best_auc:
        best_auc = mean_auc
        best_state = model.state_dict().copy()
        torch.save(best_state, "./working/best_model.pth")

# Load best model
model.load_state_dict(best_state)
print(f"\nBest Validation AUC: {best_auc:.4f}")

# Test prediction
model.eval()
test_ids = []
test_preds = []
with torch.no_grad():
    for images, ids in test_loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.sigmoid(outputs).cpu().numpy()
        test_preds.append(probs)
        test_ids.extend(ids)
test_preds = np.concatenate(test_preds, axis=0)

# Create submission
sub_df = pd.DataFrame({
    'image_id': test_ids,
    'healthy': test_preds[:, 0],
    'multiple_diseases': test_preds[:, 1],
    'rust': test_preds[:, 2],
    'scab': test_preds[:, 3]
})
sub_df = sub_df[['image_id', 'healthy', 'multiple_diseases', 'rust', 'scab']]

os.makedirs("./submission", exist_ok=True)
sub_df.to_csv("./submission/submission.csv", index=False)
print("Submission saved to ./submission/submission.csv")