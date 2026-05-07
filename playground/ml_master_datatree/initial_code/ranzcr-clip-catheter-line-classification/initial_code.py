import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms as T
import timm
from PIL import Image
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Configuration
DATA_DIR = "./input"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_DIR = os.path.join(DATA_DIR, "test")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
SUBMISSION_PATH = "./submission/submission.csv"
WORKING_DIR = "./working"

os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

SEED = 42
IMG_SIZE = 384
BATCH_SIZE = 32
EPOCHS = 8
FOLDS = 5
USE_AMP = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)
np.random.seed(SEED)

# Load data
train_df = pd.read_csv(TRAIN_CSV)
target_cols = [
    "ETT - Abnormal",
    "ETT - Borderline",
    "ETT - Normal",
    "NGT - Abnormal",
    "NGT - Borderline",
    "NGT - Incompletely Imaged",
    "NGT - Normal",
    "CVC - Abnormal",
    "CVC - Borderline",
    "CVC - Normal",
    "Swan Ganz Catheter Present",
]
print(f"Training samples: {len(train_df)}, Target columns: {len(target_cols)}")

# Patient-wise stratified split
gkf = GroupKFold(n_splits=FOLDS)
train_df["fold"] = -1
for fold, (train_idx, val_idx) in enumerate(
    gkf.split(train_df, groups=train_df["PatientID"])
):
    train_df.loc[val_idx, "fold"] = fold


# Fixed dataset class
class ChestXRayDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, is_train=True):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.is_train = is_train
        self.has_targets = all(col in df.columns for col in target_cols)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, f"{row['StudyInstanceUID']}.jpg")
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.has_targets:
            labels = row[target_cols].values.astype(np.float32)
        else:
            # For test data, return dummy labels
            labels = np.zeros(len(target_cols), dtype=np.float32)

        return image, torch.tensor(labels)


# Transformations
train_transform = T.Compose(
    [
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=10),
        T.ColorJitter(brightness=0.1, contrast=0.1),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

val_transform = T.Compose(
    [
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# Model
class EfficientNetModel(nn.Module):
    def __init__(self, num_classes=11):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b3", pretrained=True, num_classes=0
        )
        in_features = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


# Training function
def train_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        with autocast(enabled=USE_AMP):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)


# Validation function
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            all_preds.append(torch.sigmoid(outputs).cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # Compute AUC per class
    auc_scores = []
    for i in range(all_labels.shape[1]):
        try:
            auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
            auc_scores.append(auc)
        except ValueError:
            auc_scores.append(0.5)

    avg_auc = np.mean(auc_scores)
    return running_loss / len(loader.dataset), avg_auc, auc_scores


# Cross-validation training
fold_aucs = []
for fold in range(FOLDS):
    print(f"\n=== Fold {fold} ===")

    # Data split
    train_fold = train_df[train_df["fold"] != fold]
    val_fold = train_df[train_df["fold"] == fold]

    # Class weights for imbalance
    pos_counts = train_fold[target_cols].sum()
    total = len(train_fold)
    class_weights = (total - pos_counts) / (pos_counts + 1e-7)
    class_weights = torch.tensor(class_weights.values, dtype=torch.float32).to(DEVICE)

    # Datasets and loaders
    train_dataset = ChestXRayDataset(
        train_fold, TRAIN_DIR, transform=train_transform, is_train=True
    )
    val_dataset = ChestXRayDataset(
        val_fold, TRAIN_DIR, transform=val_transform, is_train=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Model, loss, optimizer
    model = EfficientNetModel(num_classes=len(target_cols)).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = GradScaler() if USE_AMP else None

    best_auc = 0.0
    for epoch in range(EPOCHS):
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, scaler, DEVICE
        )
        val_loss, val_auc, _ = validate(model, val_loader, criterion, DEVICE)
        scheduler.step()

        print(
            f"Epoch {epoch+1}/{EPOCHS}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(
                model.state_dict(), os.path.join(WORKING_DIR, f"best_fold{fold}.pth")
            )

    fold_aucs.append(best_auc)
    print(f"Fold {fold} Best AUC: {best_auc:.4f}")

print(f"\nCross-validation AUCs: {fold_aucs}")
print(f"Mean CV AUC: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")

# Ensemble predictions from all folds
print("\nGenerating ensemble predictions on test set...")
test_files = [f for f in os.listdir(TEST_DIR) if f.endswith(".jpg")]
test_df = pd.DataFrame(
    {"StudyInstanceUID": [f.replace(".jpg", "") for f in test_files]}
)

# Ensure test_df is in the same order as sample submission
sample_submission = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
test_df = test_df.merge(
    sample_submission[["StudyInstanceUID"]], on="StudyInstanceUID", how="right"
)

test_dataset = ChestXRayDataset(
    test_df, TEST_DIR, transform=val_transform, is_train=False
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=4
)

# Ensemble predictions from all folds
all_fold_preds = []
for fold in range(FOLDS):
    print(f"Loading fold {fold} model...")
    model = EfficientNetModel(num_classes=len(target_cols)).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(WORKING_DIR, f"best_fold{fold}.pth")))
    model.eval()

    fold_preds = []
    with torch.no_grad():
        for images, _ in tqdm(test_loader, desc=f"Inference fold {fold}"):
            images = images.to(DEVICE)
            outputs = model(images)
            preds = torch.sigmoid(outputs).cpu().numpy()
            fold_preds.append(preds)

    fold_preds = np.concatenate(fold_preds)
    all_fold_preds.append(fold_preds)

# Average predictions from all folds
ensemble_preds = np.mean(all_fold_preds, axis=0)

# Create submission in the correct format
submission = pd.DataFrame(ensemble_preds, columns=target_cols)
submission.insert(0, "StudyInstanceUID", test_df["StudyInstanceUID"].values)

# Ensure all columns are present and in correct order
required_columns = ["StudyInstanceUID"] + target_cols
submission = submission[required_columns]

submission.to_csv(SUBMISSION_PATH, index=False)
print(f"Submission saved to {SUBMISSION_PATH}")
print(f"Submission shape: {submission.shape}")
print("\nFirst few rows of submission:")
print(submission.head())

# Compute validation metric on the last fold
print("\nComputing final validation metric...")
val_fold = train_df[train_df["fold"] == FOLDS - 1]
val_dataset = ChestXRayDataset(
    val_fold, TRAIN_DIR, transform=val_transform, is_train=False
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=4
)

# Load the last fold model for validation
model = EfficientNetModel(num_classes=len(target_cols)).to(DEVICE)
model.load_state_dict(torch.load(os.path.join(WORKING_DIR, f"best_fold{FOLDS-1}.pth")))

# Use class weights from last fold
pos_counts = val_fold[target_cols].sum()
total = len(val_fold)
class_weights = (total - pos_counts) / (pos_counts + 1e-7)
class_weights = torch.tensor(class_weights.values, dtype=torch.float32).to(DEVICE)
criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)

val_loss, val_auc, auc_scores = validate(model, val_loader, criterion, DEVICE)
print(f"\nFinal Validation AUC: {val_auc:.4f}")
print("Per-class AUC scores:")
for col, score in zip(target_cols, auc_scores):
    print(f"  {col}: {score:.4f}")

print("\nDone! The submission.csv has been generated with ensemble predictions.")
