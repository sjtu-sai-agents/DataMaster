import os
import random
import time
import warnings
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from sklearn.model_selection import train_test_split
from torchvision import models

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
SEED = 42
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(SEED)

# Audio parameters
SR = 44100
DURATION = 4
TARGET_LEN = SR * DURATION
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
FMIN = 0
FMAX = None
POWER = 2

# Training parameters
BATCH_SIZE = 80
VAL_BATCH_SIZE = 32
NUM_WORKERS = 4
TTA_CROPS = 5
EPOCHS_COMBINED = 20
EPOCHS_FINETUNE = 15
LR_COMBINED = 1e-3
LR_FINETUNE = 1e-4
MIXUP_ALPHA = 0.4
SPEC_AUG = True
NUM_MASK = 2
FREQ_MASK_MAX = 24
TIME_MASK_MAX = 80

# Paths
INPUT_DIR = "./input"
TRAIN_CURATED_DIR = os.path.join(INPUT_DIR, "train_curated")
TRAIN_NOISY_DIR = os.path.join(INPUT_DIR, "train_noisy")
TEST_DIR = os.path.join(INPUT_DIR, "test")
CURATED_CSV = os.path.join(INPUT_DIR, "train_curated.csv")
NOISY_CSV = os.path.join(INPUT_DIR, "train_noisy.csv")
SAMPLE_SUB = os.path.join(INPUT_DIR, "sample_submission.csv")
SUBMISSION_PATH = "./submission/submission.csv"
os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
CHECKPOINT_DIR = "./working/checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ----------------------------------------------------------------------
# Label mapping
# ----------------------------------------------------------------------
sub_sample = pd.read_csv(SAMPLE_SUB)
label_columns = [col for col in sub_sample.columns if col != "fname"]
NUM_CLASSES = len(label_columns)
label2id = {lab: i for i, lab in enumerate(label_columns)}
id2label = {i: lab for i, lab in enumerate(label_columns)}

def labels_to_vector(labels_str):
    vec = np.zeros(NUM_CLASSES, dtype=np.float32)
    if isinstance(labels_str, str):
        for lab in labels_str.split(','):
            lab = lab.strip()
            if lab in label2id:
                vec[label2id[lab]] = 1.0
            else:
                print(f"Warning: label '{lab}' not in vocabulary")
    return vec

# ----------------------------------------------------------------------
# Data preparation
# ----------------------------------------------------------------------
# Curated data
curated_df = pd.read_csv(CURATED_CSV)
corrupted_files = ["1d44b0bd.wav"]
curated_df = curated_df[~curated_df["fname"].isin(corrupted_files)].copy()
curated_df["label_vec"] = curated_df["labels"].apply(labels_to_vector)
curated_df["full_path"] = curated_df["fname"].apply(lambda x: os.path.join(TRAIN_CURATED_DIR, x))

# Noisy data
noisy_df = pd.read_csv(NOISY_CSV)
noisy_df["label_vec"] = noisy_df["labels"].apply(labels_to_vector)
noisy_df["full_path"] = noisy_df["fname"].apply(lambda x: os.path.join(TRAIN_NOISY_DIR, x))

# Split curated into train/validation (85/15)
curated_train_df, curated_val_df = train_test_split(
    curated_df, test_size=0.15, random_state=SEED, shuffle=True
)

# Combined dataset for stage 1
combined_df = pd.concat([curated_train_df, noisy_df], ignore_index=True)

# Test data
test_df = pd.read_csv(SAMPLE_SUB)
test_df = test_df[["fname"]].copy()
test_df["full_path"] = test_df["fname"].apply(lambda x: os.path.join(TEST_DIR, x))

# ----------------------------------------------------------------------
# Audio processing & dataset
# ----------------------------------------------------------------------
def spec_augment(spec, num_mask=NUM_MASK, freq_masking_max=FREQ_MASK_MAX, time_masking_max=TIME_MASK_MAX):
    """Apply SpecAugment to a spectrogram tensor of shape (C, H, W)."""
    spec = spec.clone()
    for _ in range(num_mask):
        # Frequency masking
        f = random.randint(0, freq_masking_max)
        f0 = random.randint(0, spec.shape[1] - f)
        spec[:, f0:f0+f, :] = 0
        # Time masking
        t = random.randint(0, time_masking_max)
        t0 = random.randint(0, spec.shape[2] - t)
        spec[:, :, t0:t0+t] = 0
    return spec

class AudioDataset(Dataset):
    def __init__(self, df, mode="train", tta_crops=1, spec_aug=False):
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.tta_crops = tta_crops if mode != "train" else 1
        self.spec_aug = spec_aug and mode == "train"
        self.sr = SR
        self.target_len = TARGET_LEN
        self.labels_available = "label_vec" in df.columns

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filepath = row["full_path"]

        # Load audio
        y, _ = librosa.load(filepath, sr=self.sr, mono=True)
        y = y.astype(np.float32)

        # Determine start indices for crops
        starts = self._get_start_indices(len(y), self.tta_crops)

        crops = []
        for start in starts:
            if len(y) <= self.target_len:
                # Pad to target length
                if len(y) < self.target_len:
                    pad_len = self.target_len - len(y)
                    y_pad = np.pad(y, (0, pad_len), mode="constant")
                else:
                    y_pad = y
                crop = y_pad[:self.target_len]
            else:
                end = start + self.target_len
                crop = y[start:end]
                if len(crop) < self.target_len:
                    crop = np.pad(crop, (0, self.target_len - len(crop)), mode="constant")

            # Compute Mel + deltas
            features = self._compute_features(crop)
            # Instance normalization
            features = self._normalize(features)
            tensor = torch.from_numpy(features).float()
            if self.spec_aug:
                tensor = spec_augment(tensor)
            crops.append(tensor)

        # Stack crops if more than one
        if len(crops) > 1:
            data = torch.stack(crops, dim=0)  # (num_crops, C, H, W)
        else:
            data = crops[0]  # (C, H, W)

        if self.labels_available:
            label_vec = row["label_vec"]
            label = torch.from_numpy(label_vec).float()
            return data, label
        else:
            return data, row["fname"]

    def _get_start_indices(self, len_y, num_crops):
        if len_y <= self.target_len:
            return [0] * num_crops
        max_start = len_y - self.target_len
        if self.mode == "train":
            # Single random crop for training
            return [random.randint(0, max_start)]
        else:
            # Equally spaced crops for validation/test
            if num_crops == 1:
                return [max_start // 2]  # center crop
            step = max_start / (num_crops - 1)
            starts = [int(round(i * step)) for i in range(num_crops)]
            if starts[-1] > max_start:
                starts[-1] = max_start
            return starts

    def _compute_features(self, y):
        # y length is exactly target_len
        mel = librosa.feature.melspectrogram(
            y=y, sr=self.sr, n_fft=N_FFT, hop_length=HOP_LENGTH,
            n_mels=N_MELS, power=POWER, fmin=FMIN, fmax=FMAX
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        delta = librosa.feature.delta(mel_db)
        delta2 = librosa.feature.delta(mel_db, order=2)
        features = np.stack([mel_db, delta, delta2], axis=0)  # (3, H, W)
        return features

    def _normalize(self, features):
        for c in range(features.shape[0]):
            mean = features[c].mean()
            std = features[c].std()
            if std > 0:
                features[c] = (features[c] - mean) / std
            else:
                features[c] = features[c] - mean
        return features

# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------
def get_model():
    try:
        model = models.resnext50_32x4d(pretrained=True)
    except:
        model = models.resnet50(pretrained=True)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, NUM_CLASSES)
    return model

# ----------------------------------------------------------------------
# Mixup
# ----------------------------------------------------------------------
def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

# ----------------------------------------------------------------------
# Evaluation metric: label-weighted label-ranking average precision
# ----------------------------------------------------------------------
def lwlrap(truth, scores):
    """
    truth : np.array of shape (n_samples, n_labels) with 0/1
    scores: np.array of shape (n_samples, n_labels) with confidence values
    Returns overall lwlrap and per-class values.
    """
    n_samples, n_labels = truth.shape
    per_class_lrap = np.zeros(n_labels)
    for label_idx in range(n_labels):
        scores_label = scores[:, label_idx]
        truth_label = truth[:, label_idx]
        order = np.argsort(scores_label)[::-1]
        truth_label = truth_label[order]
        tp = np.cumsum(truth_label)
        rank = np.arange(1, n_samples + 1)
        prec = tp / rank
        if np.sum(truth_label) > 0:
            per_class_lrap[label_idx] = np.sum(prec * truth_label) / np.sum(truth_label)
        else:
            per_class_lrap[label_idx] = 0.0
    support = np.sum(truth, axis=0)
    overall_lwlrap = np.sum(per_class_lrap * support) / np.sum(support)
    return overall_lwlrap, per_class_lrap

# ----------------------------------------------------------------------
# Training and evaluation utilities
# ----------------------------------------------------------------------
def train_one_epoch(model, optimizer, loss_fn, loader, device, scaler, mixup_alpha):
    model.train()
    running_loss = 0.0
    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        if mixup_alpha > 0:
            inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, mixup_alpha)
            with autocast():
                outputs = model(inputs)
                loss = lam * loss_fn(outputs, labels_a) + (1 - lam) * loss_fn(outputs, labels_b)
        else:
            with autocast():
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(loader.dataset)

def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in loader:
            # inputs may be (batch, C, H, W) or (batch, crops, C, H, W)
            if len(inputs.shape) == 5:
                batch_size, num_crops, C, H, W = inputs.shape
                inputs = inputs.view(-1, C, H, W).to(device)
                outputs = model(inputs)
                outputs = outputs.view(batch_size, num_crops, -1)
                outputs = outputs.mean(dim=1)
            else:
                inputs = inputs.to(device)
                outputs = model(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_preds.append(probs)
            all_labels.append(labels.numpy())
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    overall, _ = lwlrap(all_labels, all_preds)
    return overall

def train_model(model, train_loader, val_loader, epochs, lr, mixup_alpha, checkpoint_path, device):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    loss_fn = nn.BCEWithLogitsLoss()
    scaler = GradScaler()
    best_lwlrap = 0.0
    for epoch in range(1, epochs+1):
        start_time = time.time()
        train_loss = train_one_epoch(model, optimizer, loss_fn, train_loader, device, scaler, mixup_alpha)
        scheduler.step()
        val_lwlrap = evaluate(model, val_loader, device)
        elapsed = time.time() - start_time
        print(f"Epoch {epoch:2d}/{epochs} | train loss: {train_loss:.4f} | val lwlrap: {val_lwlrap:.6f} | time: {elapsed:.0f}s")
        if val_lwlrap > best_lwlrap:
            best_lwlrap = val_lwlrap
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  -> New best model saved (lwlrap {best_lwlrap:.6f})")
    return best_lwlrap

# ----------------------------------------------------------------------
# DataLoaders
# ----------------------------------------------------------------------
def worker_init_fn(worker_id):
    set_seed(SEED + worker_id)

# Validation dataset (always from curated_val, with TTA)
val_dataset = AudioDataset(curated_val_df, mode="val", tta_crops=TTA_CROPS, spec_aug=False)
val_loader = DataLoader(
    val_dataset,
    batch_size=VAL_BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    worker_init_fn=worker_init_fn
)

# ----------------------------------------------------------------------
# Stage 1: Training on combined data
# ----------------------------------------------------------------------
print("\n" + "="*60)
print("Stage 1: Training on combined (curated+noisy) data")
print("="*60)

train_dataset_combined = AudioDataset(combined_df, mode="train", spec_aug=SPEC_AUG)
train_loader_combined = DataLoader(
    train_dataset_combined,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    drop_last=True,
    worker_init_fn=worker_init_fn
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model().to(device)

stage1_checkpoint = os.path.join(CHECKPOINT_DIR, "stage1_best.pth")
best_stage1_lwlrap = train_model(
    model, train_loader_combined, val_loader,
    epochs=EPOCHS_COMBINED,
    lr=LR_COMBINED,
    mixup_alpha=MIXUP_ALPHA,
    checkpoint_path=stage1_checkpoint,
    device=device
)
print(f"Stage 1 best validation lwlrap: {best_stage1_lwlrap:.6f}")

# ----------------------------------------------------------------------
# Stage 2: Fine-tuning on curated data only
# ----------------------------------------------------------------------
print("\n" + "="*60)
print("Stage 2: Fine-tuning on curated data only")
print("="*60)

# Reload best model from stage 1
model.load_state_dict(torch.load(stage1_checkpoint, map_location=device))

curated_train_dataset = AudioDataset(curated_train_df, mode="train", spec_aug=SPEC_AUG)
curated_train_loader = DataLoader(
    curated_train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    drop_last=True,
    worker_init_fn=worker_init_fn
)

stage2_checkpoint = os.path.join(CHECKPOINT_DIR, "stage2_best.pth")
best_stage2_lwlrap = train_model(
    model, curated_train_loader, val_loader,
    epochs=EPOCHS_FINETUNE,
    lr=LR_FINETUNE,
    mixup_alpha=MIXUP_ALPHA,
    checkpoint_path=stage2_checkpoint,
    device=device
)
print(f"Stage 2 best validation lwlrap: {best_stage2_lwlrap:.6f}")

# Load the best model from stage 2 for final evaluation and prediction
model.load_state_dict(torch.load(stage2_checkpoint, map_location=device))

# ----------------------------------------------------------------------
# Final validation metric
# ----------------------------------------------------------------------
final_val_lwlrap = evaluate(model, val_loader, device)
print(f"\nFinal validation lwlrap: {final_val_lwlrap:.6f}")

# ----------------------------------------------------------------------
# Generate submission on test set
# ----------------------------------------------------------------------
print("\nGenerating test predictions...")
test_dataset = AudioDataset(test_df, mode="test", tta_crops=TTA_CROPS, spec_aug=False)
test_loader = DataLoader(
    test_dataset,
    batch_size=VAL_BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    worker_init_fn=worker_init_fn
)

model.eval()
all_preds = []
all_fnames = []
with torch.no_grad():
    for inputs, fnames in test_loader:
        if len(inputs.shape) == 5:
            batch_size, num_crops, C, H, W = inputs.shape
            inputs = inputs.view(-1, C, H, W).to(device)
            outputs = model(inputs)
            outputs = outputs.view(batch_size, num_crops, -1)
            outputs = outputs.mean(dim=1)
        else:
            inputs = inputs.to(device)
            outputs = model(inputs)
        probs = torch.sigmoid(outputs).cpu().numpy()
        all_preds.append(probs)
        all_fnames.extend(fnames)

all_preds = np.concatenate(all_preds, axis=0)
submission = pd.DataFrame(all_preds, columns=label_columns)
submission.insert(0, "fname", all_fnames)
submission.to_csv(SUBMISSION_PATH, index=False)
print(f"Submission saved to {SUBMISSION_PATH}")