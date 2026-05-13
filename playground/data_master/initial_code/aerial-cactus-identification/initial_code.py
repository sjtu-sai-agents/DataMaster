import os
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

# Set random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True

# Paths
INPUT_DIR = "./input"
TRAIN_IMG_DIR = os.path.join(INPUT_DIR, "train")
TEST_IMG_DIR = os.path.join(INPUT_DIR, "test")
TRAIN_CSV = os.path.join(INPUT_DIR, "train.csv")
SAMPLE_SUB = os.path.join(INPUT_DIR, "sample_submission.csv")
OUTPUT_DIR = "./submission"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Hyperparameters
BATCH_SIZE = 32
IMG_SIZE = 224
LR = 0.0001
MAX_EPOCHS = 15
PATIENCE = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load training labels
train_df = pd.read_csv(TRAIN_CSV)

# Split into train and validation (90/10 stratified)
train_ids, val_ids = train_test_split(
    train_df.index,
    test_size=0.1,
    stratify=train_df['has_cactus'],
    random_state=SEED
)
train_df_sub = train_df.loc[train_ids].reset_index(drop=True)
val_df = train_df.loc[val_ids].reset_index(drop=True)

# ImageNet normalization
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Transforms
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# Test transforms for TTA
test_transform_orig = val_transform
test_transform_hflip = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=1.0),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
test_transform_vflip = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomVerticalFlip(p=1.0),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# Dataset class
class CactusDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, is_test=False):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.df.iloc[idx]['id']
        img_path = os.path.join(self.img_dir, img_id)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        if self.is_test:
            return image, img_id
        else:
            label = self.df.iloc[idx]['has_cactus']
            return image, torch.tensor(label, dtype=torch.float)

# Create datasets and dataloaders
train_dataset = CactusDataset(train_df_sub, TRAIN_IMG_DIR, train_transform)
val_dataset = CactusDataset(val_df, TRAIN_IMG_DIR, val_transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# Model
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 1)
model = model.to(DEVICE)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# Training loop with early stopping
best_auc = 0.0
best_epoch = 0
patience_counter = 0
best_model_path = os.path.join(OUTPUT_DIR, "best_model.pth")

for epoch in range(MAX_EPOCHS):
    model.train()
    train_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE).view(-1, 1)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)
    train_loss /= len(train_loader.dataset)

    # Validation
    model.eval()
    val_preds = []
    val_labels = []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            val_preds.extend(probs)
            val_labels.extend(labels.cpu().numpy())
    auc = roc_auc_score(val_labels, val_preds)
    print(f"Epoch {epoch+1}/{MAX_EPOCHS} - Train Loss: {train_loss:.4f} - Val AUC: {auc:.4f}")

    # Early stopping and checkpoint
    if auc > best_auc:
        best_auc = auc
        best_epoch = epoch
        patience_counter = 0
        torch.save(model.state_dict(), best_model_path)
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break

print(f"\nBest validation AUC: {best_auc:.6f} (epoch {best_epoch+1})")

# Load best model for test predictions
model.load_state_dict(torch.load(best_model_path))
model.eval()

# Load test IDs from sample submission
sub_df = pd.read_csv(SAMPLE_SUB)
test_ids = sub_df['id'].tolist()
test_df = pd.DataFrame({'id': test_ids})

# Create test datasets for TTA
test_dataset_orig = CactusDataset(test_df, TEST_IMG_DIR, test_transform_orig, is_test=True)
test_dataset_hflip = CactusDataset(test_df, TEST_IMG_DIR, test_transform_hflip, is_test=True)
test_dataset_vflip = CactusDataset(test_df, TEST_IMG_DIR, test_transform_vflip, is_test=True)

test_loader_orig = DataLoader(test_dataset_orig, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
test_loader_hflip = DataLoader(test_dataset_hflip, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
test_loader_vflip = DataLoader(test_dataset_vflip, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

def predict_tta(loader):
    all_probs = []
    all_ids = []
    with torch.no_grad():
        for images, ids in loader:
            images = images.to(DEVICE)
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            all_probs.extend(probs)
            all_ids.extend(ids)
    # Ensure order matches original test_df
    df = pd.DataFrame({'id': all_ids, 'prob': all_probs})
    df = df.set_index('id').reindex(test_ids).reset_index()  # align order
    return df['prob'].values

probs_orig = predict_tta(test_loader_orig)
probs_hflip = predict_tta(test_loader_hflip)
probs_vflip = predict_tta(test_loader_vflip)

# Average probabilities
final_probs = (probs_orig + probs_hflip + probs_vflip) / 3.0

# Create submission file
submission = pd.DataFrame({'id': test_ids, 'has_cactus': final_probs})
submission.to_csv(os.path.join(OUTPUT_DIR, "submission.csv"), index=False)
print("Submission saved to submission/submission.csv")