import os
import re
import glob
import random
import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

# tqdm fallback
try:
    from tqdm.auto import tqdm
except ImportError:
    class tqdm:
        def __init__(self, iterable=None, desc=None, total=None):
            self.iterable = iterable
            self.desc = desc
            if total is None:
                total = len(iterable)
            self.total = total
        def __iter__(self):
            if self.desc:
                print(self.desc)
            for i, item in enumerate(self.iterable):
                if i % 10 == 0:
                    print(f"Processed {i}/{self.total}")
                yield item
            print(f"Processed {self.total}/{self.total}")
        def update(self, n=1):
            pass

# Set reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ------------------ Helper functions ------------------
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def read_dicom_pixels(path):
    try:
        dcm = pydicom.dcmread(path, force=True)
        img = dcm.pixel_array.astype(np.float32)
        if hasattr(dcm, 'RescaleSlope') and hasattr(dcm, 'RescaleIntercept'):
            img = img * dcm.RescaleSlope + dcm.RescaleIntercept
        return img
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None

def normalize_image(img):
    if img.size == 0:
        return np.zeros_like(img, dtype=np.float32)
    non_zero = img[img != 0]
    if len(non_zero) < 10:
        lo = np.percentile(img, 1)
        hi = np.percentile(img, 99)
    else:
        lo = np.percentile(non_zero, 1)
        hi = np.percentile(non_zero, 99)
    if hi <= lo:
        lo = np.min(img)
        hi = np.max(img)
        if hi <= lo:
            return np.zeros_like(img, dtype=np.float32)
    clipped = np.clip(img, lo, hi)
    normed = (clipped - lo) / (hi - lo + 1e-6)
    return normed.astype(np.float32)

def resize2d(img, out_h=224, out_w=224):
    tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    tensor = F.interpolate(tensor, size=(out_h, out_w), mode='bilinear', align_corners=False)
    return tensor.squeeze().numpy()

def preprocess_subject(base_dir, subject_id, modalities=['FLAIR','T1wCE','T2w'], size=224):
    subject_dir = os.path.join(base_dir, subject_id)
    slices = []
    for mod in modalities:
        mod_dir = os.path.join(subject_dir, mod)
        if not os.path.isdir(mod_dir):
            slices.append(np.zeros((size,size), dtype=np.float32))
            continue
        files = sorted(glob.glob(os.path.join(mod_dir, '*.dcm')), key=natural_sort_key)
        if not files:
            slices.append(np.zeros((size,size), dtype=np.float32))
            continue
        mid_idx = len(files) // 2
        dcm_path = files[mid_idx]
        img = read_dicom_pixels(dcm_path)
        if img is None:
            slices.append(np.zeros((size,size), dtype=np.float32))
            continue
        normed = normalize_image(img)
        resized = resize2d(normed, size, size)
        slices.append(resized)
    stack = np.stack(slices, axis=0)  # (3, size, size)
    return stack.astype(np.float32)

def cache_images(base_dir, ids, size=224):
    cache = {}
    for subj in tqdm(ids, desc=f"Caching from {base_dir}"):
        cache[subj] = preprocess_subject(base_dir, subj, size=size)
    return cache

# ------------------ Dataset ------------------
class SubjectImageDataset(Dataset):
    def __init__(self, ids, cache, labels_dict=None, transform=False):
        self.ids = ids
        self.cache = cache
        self.labels_dict = labels_dict
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        subj = self.ids[idx]
        img = self.cache[subj]  # (C, H, W) numpy
        img = torch.from_numpy(img).float().contiguous()
        if self.transform:
            if random.random() < 0.5:
                img = img.flip(-1)   # horizontal
            if random.random() < 0.5:
                img = img.flip(-2)   # vertical
        if self.labels_dict is not None:
            label = self.labels_dict[subj]
            return img, torch.tensor(label, dtype=torch.float32)
        else:
            return img

# ------------------ Model ------------------
class SimpleCNN(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(256, 1)

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# ------------------ Utility ------------------
def safe_roc_auc(y_true, y_pred):
    if len(np.unique(y_true)) == 1:
        return 0.5
    else:
        return roc_auc_score(y_true, y_pred)

# ------------------ Training per fold ------------------
def train_fold(fold, train_ids, val_ids, cache, labels_dict, device, epochs=6, batch_size=16):
    train_dataset = SubjectImageDataset(train_ids, cache, labels_dict, transform=True)
    val_dataset = SubjectImageDataset(val_ids, cache, labels_dict, transform=False)

    num_workers = max(2, os.cpu_count() // 2)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    model = SimpleCNN().to(device)

    # class-balanced positive weight
    train_labels = [labels_dict[i] for i in train_ids]
    pos = sum(train_labels)
    neg = len(train_labels) - pos
    pos_weight = torch.tensor([neg / max(pos, 1.0)], device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_auc = 0.0
    best_model_state = None

    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_dataset)

        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device, non_blocking=True)
                outputs = model(inputs)
                probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                val_preds.extend(probs)
                val_true.extend(targets.cpu().numpy())
        auc = safe_roc_auc(val_true, val_preds)
        print(f'Fold {fold} Epoch {epoch} - Loss: {epoch_loss:.4f} Val AUC: {auc:.4f}')
        if auc > best_auc:
            best_auc = auc
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}

    # Load best model and get final validation predictions
    model.load_state_dict(best_model_state)
    model.eval()
    val_preds, val_true = [], []
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            val_preds.extend(probs)
            val_true.extend(targets.cpu().numpy())
    return val_preds, val_true, model, best_auc

# ------------------ Main ------------------
def main():
    input_dir = "./input"
    train_dir = os.path.join(input_dir, "train")
    test_dir = os.path.join(input_dir, "test")

    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Train directory not found: {train_dir}")
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    # Load metadata
    train_df = pd.read_csv(os.path.join(input_dir, "train_labels.csv"))
    sample_df = pd.read_csv(os.path.join(input_dir, "sample_submission.csv"))

    # Prepare IDs (padded for file access, raw for submission)
    train_ids = [f"{i:05d}" for i in train_df['BraTS21ID']]
    train_labels = train_df['MGMT_value'].values
    test_ids_padded = [f"{i:05d}" for i in sample_df['BraTS21ID']]
    test_ids_raw = sample_df['BraTS21ID'].tolist()

    # Exclude problematic cases
    exclude = ['00109', '00123', '00709']
    keep_idx = [i for i, sid in enumerate(train_ids) if sid not in exclude]
    train_ids = [train_ids[i] for i in keep_idx]
    train_labels = [train_labels[i] for i in keep_idx]

    labels_dict = {sid: lbl for sid, lbl in zip(train_ids, train_labels)}

    # Cache images
    print("Caching training images...")
    train_cache = cache_images(train_dir, train_ids, size=224)
    print("Caching test images...")
    test_cache = cache_images(test_dir, test_ids_padded, size=224)

    cache = {**train_cache, **test_cache}

    # Cross‑validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_pred = np.zeros(len(train_ids))
    oof_true = np.zeros(len(train_ids))
    test_preds = np.zeros(len(test_ids_padded))

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_ids, train_labels), start=1):
        print(f"\n===== Fold {fold} =====")
        fold_train_ids = [train_ids[i] for i in train_idx]
        fold_val_ids = [train_ids[i] for i in val_idx]

        val_preds, val_true, model, best_auc = train_fold(
            fold, fold_train_ids, fold_val_ids, cache, labels_dict, device
        )
        oof_pred[val_idx] = val_preds
        oof_true[val_idx] = val_true
        print(f"Fold {fold} Best AUC: {best_auc:.4f}")

        # Test predictions for this fold
        test_dataset = SubjectImageDataset(test_ids_padded, cache, labels_dict=None, transform=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,
                                 num_workers=max(2, os.cpu_count()//2), pin_memory=True)
        fold_test_preds = []
        model.to(device)
        model.eval()
        with torch.no_grad():
            for inputs in test_loader:
                inputs = inputs.to(device, non_blocking=True)
                outputs = model(inputs)
                probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                fold_test_preds.append(probs)
        fold_test_preds = np.concatenate(fold_test_preds)
        test_preds += fold_test_preds / 5.0  # average

    # Overall validation metric
    oof_auc = safe_roc_auc(oof_true, oof_pred)
    print(f"\nOverall Out-of-Fold AUC: {oof_auc:.4f}")

    # Sanity check and submission
    if np.isnan(test_preds).any():
        test_preds = np.nan_to_num(test_preds, nan=np.mean(train_labels))

    os.makedirs('./submission', exist_ok=True)
    submission = pd.DataFrame({'BraTS21ID': test_ids_raw, 'MGMT_value': test_preds})
    submission.to_csv('./submission/submission.csv', index=False)
    print("Submission saved to ./submission/submission.csv")

if __name__ == "__main__":
    main()