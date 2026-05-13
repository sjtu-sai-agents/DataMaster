import os
import random
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import timm
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from contextlib import nullcontext

# Reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42)

# Directories
INPUT_DIR = "./input"
TRAIN_DIR = os.path.join(INPUT_DIR, "train")
TEST_DIR = os.path.join(INPUT_DIR, "test")
SUBMISSION_DIR = "./submission"
WORKING_DIR = "./working"
os.makedirs(SUBMISSION_DIR, exist_ok=True)
os.makedirs(WORKING_DIR, exist_ok=True)

# ------------------------------------------------------------
# Load training data
# ------------------------------------------------------------
train_files, train_labels = [], []
for fname in os.listdir(TRAIN_DIR):
    if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        label = 1 if fname.lower().startswith('dog') else 0
        train_files.append(os.path.join(TRAIN_DIR, fname))
        train_labels.append(label)
train_files = np.array(train_files)
train_labels = np.array(train_labels)
print(f"Found {len(train_files)} training images.")

# ------------------------------------------------------------
# Load test IDs (order from sample submission)
# ------------------------------------------------------------
sample_sub_path = os.path.join(INPUT_DIR, "sample_submission.csv")
if os.path.exists(sample_sub_path):
    sample_df = pd.read_csv(sample_sub_path)
    test_ids = sample_df['id'].values
    test_files = [os.path.join(TEST_DIR, f"{i}.jpg") for i in test_ids]
    # verify existence
    missing = [f for f in test_files if not os.path.exists(f)]
    if missing:
        print(f"Warning: {len(missing)} test files missing, fallback to scanning directory.")
        test_files, test_ids = [], []
        for fname in os.listdir(TEST_DIR):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                try:
                    iid = int(os.path.splitext(fname)[0])
                except:
                    continue
                test_ids.append(iid)
                test_files.append(os.path.join(TEST_DIR, fname))
        sort_idx = np.argsort(test_ids)
        test_ids = np.array(test_ids)[sort_idx]
        test_files = np.array(test_files)[sort_idx]
else:
    print("sample_submission.csv not found, scanning test directory.")
    test_files, test_ids = [], []
    for fname in os.listdir(TEST_DIR):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            try:
                iid = int(os.path.splitext(fname)[0])
            except:
                continue
            test_ids.append(iid)
            test_files.append(os.path.join(TEST_DIR, fname))
    sort_idx = np.argsort(test_ids)
    test_ids = np.array(test_ids)[sort_idx]
    test_files = np.array(test_files)[sort_idx]
print(f"Found {len(test_files)} test images.")

# ------------------------------------------------------------
# Train / Validation stratified split (80/20)
# ------------------------------------------------------------
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(sss.split(train_files, train_labels))
tr_files, tr_labels = train_files[train_idx], train_labels[train_idx]
va_files, va_labels = train_files[val_idx], train_labels[val_idx]
print(f"Train: {len(tr_files)}, Validation: {len(va_files)}")

# ------------------------------------------------------------
# Image transforms
# ------------------------------------------------------------
IMG_SIZE = 224
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]
normalize = transforms.Normalize(mean=IMG_MEAN, std=IMG_STD)

# Base transform (no augmentation)
base_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    normalize
])

# Always flip transform (augmentation)
hflip_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(p=1.0),
    transforms.ToTensor(),
    normalize
])

# TenCrop transform for Test-Time Augmentation (TTA)
tencrop_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.TenCrop(IMG_SIZE),
    transforms.Lambda(lambda crops: torch.stack(
        [normalize(transforms.ToTensor()(crop)) for crop in crops]
    ))
])

# ------------------------------------------------------------
# Custom Dataset
# ------------------------------------------------------------
class ImagePathDataset(Dataset):
    def __init__(self, files, labels=None, transform=None):
        self.files = files
        self.labels = labels if labels is not None else None
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        if self.labels is not None:
            return img, int(self.labels[idx])
        else:
            return img, -1

# Create datasets
train_ds_base = ImagePathDataset(tr_files, tr_labels, transform=base_transform)
train_ds_flip = ImagePathDataset(tr_files, tr_labels, transform=hflip_transform)
val_ds_tta = ImagePathDataset(va_files, va_labels, transform=tencrop_transform)
test_ds_tta = ImagePathDataset(test_files, None, transform=tencrop_transform)

# ------------------------------------------------------------
# DataLoader setup
# ------------------------------------------------------------
def get_num_workers():
    cpu_count = os.cpu_count()
    if cpu_count is None:
        return 4
    else:
        return max(4, min(12, cpu_count - 1 if cpu_count > 1 else 4))

num_workers = get_num_workers()
print(f"DataLoader using {num_workers} workers")

batch_size_train = 128
batch_size_tta = 64   # each sample gives 10 crops

def make_loader(dataset, batch_size, shuffle=False):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        drop_last=False
    )

train_loader_base = make_loader(train_ds_base, batch_size_train)
train_loader_flip = make_loader(train_ds_flip, batch_size_train)
val_loader_tta = make_loader(val_ds_tta, batch_size_tta)
test_loader_tta = make_loader(test_ds_tta, batch_size_tta)

# ------------------------------------------------------------
# Device
# ------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------------------------------------------------
# Load pretrained model from timm
# ------------------------------------------------------------
model_names = [
    "vit_large_patch14_224.clip",
    "swin_large_patch4_window7_224",
    "convnext_large.fb_in22k_ft_in1k",
    "vit_base_patch16_224",
]

model = None
for name in model_names:
    try:
        print(f"Trying to load {name}...")
        model = timm.create_model(name, pretrained=True, num_classes=0)
        print(f"Loaded {name} successfully.")
        selected_model_name = name
        break
    except Exception as e:
        print(f"Failed to load {name}: {e}")

if model is None:
    print("All models failed, falling back to vit_base_patch16_224 without pretrained weights.")
    model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0)
    selected_model_name = "vit_base_patch16_224"

model = model.to(device)
model.eval()
print(f"Model: {selected_model_name}")

# ------------------------------------------------------------
# Chunk sizes for feature extraction (avoid OOM)
# ------------------------------------------------------------
large_models = ["vit_large_patch14_224.clip", "swin_large_patch4_window7_224", "convnext_large.fb_in22k_ft_in1k"]
if selected_model_name in large_models:
    tta_chunk = 256
    single_chunk = 512
else:
    tta_chunk = 512
    single_chunk = 1024
print(f"Chunk sizes: TTA={tta_chunk}, single={single_chunk}")

# ------------------------------------------------------------
# Feature extraction function
# ------------------------------------------------------------
def extract_features(model, loader, is_tta=False, chunk_size=None):
    model.eval()
    features = []
    autocast = torch.cuda.amp.autocast if torch.cuda.is_available() else nullcontext
    with torch.no_grad():
        for imgs, _ in loader:
            if is_tta:
                B, N_crops, C, H, W = imgs.shape
                imgs = imgs.view(B * N_crops, C, H, W).to(device, non_blocking=True)
                out_list = []
                for i in range(0, imgs.size(0), chunk_size):
                    chunk = imgs[i:i+chunk_size]
                    with autocast():
                        feat = model(chunk)
                    out_list.append(feat.cpu())
                feats = torch.cat(out_list, dim=0)          # (B*N_crops, feat_dim)
                feats = feats.view(B, N_crops, -1).mean(dim=1)  # average crops
            else:
                imgs = imgs.to(device, non_blocking=True)
                out_list = []
                for i in range(0, imgs.size(0), chunk_size):
                    chunk = imgs[i:i+chunk_size]
                    with autocast():
                        feat = model(chunk)
                    out_list.append(feat.cpu())
                feats = torch.cat(out_list, dim=0)
            features.append(feats.numpy())
    return np.vstack(features)

# ------------------------------------------------------------
# Extract features
# ------------------------------------------------------------
print("Extracting features from training base set...")
X_train_base = extract_features(model, train_loader_base, is_tta=False, chunk_size=single_chunk)
print("Extracting features from training flip set...")
X_train_flip = extract_features(model, train_loader_flip, is_tta=False, chunk_size=single_chunk)
print("Extracting features from validation TTA set...")
X_val = extract_features(model, val_loader_tta, is_tta=True, chunk_size=tta_chunk)
print("Extracting features from test TTA set...")
X_test = extract_features(model, test_loader_tta, is_tta=True, chunk_size=tta_chunk)

# Combine training features
X_train = np.vstack([X_train_base, X_train_flip])
y_train = np.concatenate([tr_labels, tr_labels])
print(f"Training features shape: {X_train.shape}, validation shape: {X_val.shape}, test shape: {X_test.shape}")

# ------------------------------------------------------------
# L2 normalization
# ------------------------------------------------------------
def l2_normalize(X):
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    norm = np.maximum(norm, 1e-12)
    return X / norm

X_train = l2_normalize(X_train)
X_val = l2_normalize(X_val)
X_test = l2_normalize(X_test)

# ------------------------------------------------------------
# Train Logistic Regression, tune C
# ------------------------------------------------------------
C_candidates = [0.25, 0.5, 1.0, 2.0, 4.0]
best_logloss = np.inf
best_C = None
best_clf = None

for C in C_candidates:
    clf = LogisticRegression(penalty='l2', C=C, solver='lbfgs', max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    val_probs = clf.predict_proba(X_val)[:, 1]
    val_probs = np.clip(val_probs, 1e-7, 1-1e-7)
    loss = log_loss(va_labels, val_probs)
    print(f"C={C}: validation log loss = {loss:.6f}")
    if loss < best_logloss:
        best_logloss = loss
        best_C = C
        best_clf = clf

print(f"Best C: {best_C} with log loss {best_logloss:.6f}")

# ------------------------------------------------------------
# Platt scaling (calibration) on validation set
# ------------------------------------------------------------
val_logits = best_clf.decision_function(X_val)   # shape (n_val,)
calib = LogisticRegression(penalty='l2', C=1e6, solver='lbfgs', max_iter=1000, random_state=42)
calib.fit(val_logits.reshape(-1, 1), va_labels)
val_cal_probs = calib.predict_proba(val_logits.reshape(-1, 1))[:, 1]
val_cal_probs = np.clip(val_cal_probs, 1e-7, 1-1e-7)
val_cal_logloss = log_loss(va_labels, val_cal_probs)
print(f"Validation log loss after calibration: {val_cal_logloss:.6f}")

# ------------------------------------------------------------
# Test predictions
# ------------------------------------------------------------
test_logits = best_clf.decision_function(X_test)
test_probs = calib.predict_proba(test_logits.reshape(-1, 1))[:, 1]
test_probs = np.clip(test_probs, 1e-7, 1-1e-7)

sub_df = pd.DataFrame({'id': test_ids, 'label': test_probs})
sub_df.to_csv(os.path.join(SUBMISSION_DIR, 'submission.csv'), index=False)
sub_df.to_csv(os.path.join(WORKING_DIR, 'submission.csv'), index=False)
print(f"Saved submission file with {len(sub_df)} predictions.")
print("Done.")