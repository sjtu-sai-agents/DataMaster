import os
import glob
import random
import numpy as np
import pandas as pd
import librosa
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import warnings
warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True  # speeds up if input sizes fixed

SAMPLE_RATE = 2000
DURATION = 3.0
N_SAMPLES = int(SAMPLE_RATE * DURATION)  # 6000
N_MELS = 64
N_FFT = 512
HOP_LENGTH = 128
FMIN = 30
FMAX = 800
TIME_FRAMES = 1 + (N_SAMPLES - N_FFT) // HOP_LENGTH  # 43
BATCH_SIZE = 256
NUM_WORKERS = min(12, max(1, os.cpu_count() - 1))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TTA_SHIFTS = [0.0, 0.25, -0.25, 0.5, -0.5]  # seconds
MAX_EPOCHS = 14
EARLY_STOP_PATIENCE = 5
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
FINETUNE_EPOCHS = 5
FINETUNE_LR = 5e-4
USE_AMP = True

# Paths
INPUT_DIR = "./input"
TRAIN_DIR = os.path.join(INPUT_DIR, "train2")
TEST_DIR = os.path.join(INPUT_DIR, "test2")
SUBMISSION_DIR = "./submission"
os.makedirs(SUBMISSION_DIR, exist_ok=True)
WORKING_DIR = "./working"
os.makedirs(WORKING_DIR, exist_ok=True)

# ----------------------------------------------------------------------
# Data: collect files and labels
# ----------------------------------------------------------------------
def get_train_files():
    """Return list of (file_path, label) for training."""
    files = glob.glob(os.path.join(TRAIN_DIR, "*.aif"))
    file_label_pairs = []
    for f in files:
        basename = os.path.basename(f)
        if basename.endswith('_1.aif'):
            label = 1
        elif basename.endswith('_0.aif'):
            label = 0
        else:
            # fallback: try to parse last token before extension
            parts = basename[:-4].split('_')
            label = int(parts[-1]) if parts[-1].isdigit() else 0
        file_label_pairs.append((f, label))
    return file_label_pairs

train_items = get_train_files()
if len(train_items) == 0:
    raise RuntimeError("No training files found.")
train_paths, train_labels = zip(*train_items)
train_paths, train_labels = list(train_paths), list(train_labels)

test_paths = sorted(glob.glob(os.path.join(TEST_DIR, "*.aif")))
if len(test_paths) == 0:
    raise RuntimeError("No test files found.")
print(f"Found {len(train_paths)} training files, {len(test_paths)} test files.")

# Stratified split
train_idx, val_idx = train_test_split(
    range(len(train_paths)),
    test_size=0.2,
    random_state=RANDOM_SEED,
    stratify=train_labels
)
train_paths_split = [train_paths[i] for i in train_idx]
train_labels_split = [train_labels[i] for i in train_idx]
val_paths = [train_paths[i] for i in val_idx]
val_labels = [train_labels[i] for i in val_idx]

print(f"Train split: {len(train_paths_split)} files, positive ratio: {sum(train_labels_split)/len(train_labels_split):.3f}")
print(f"Val split:   {len(val_paths)} files, positive ratio: {sum(val_labels)/len(val_labels):.3f}")

# ----------------------------------------------------------------------
# Audio preprocessing
# ----------------------------------------------------------------------
def load_waveform(file_path, shift_sec=0.0):
    """
    Load audio, resample to SAMPLE_RATE, fix length to N_SAMPLES.
    If shift_sec != 0, circular shift waveform by shift_samples (positive shift moves later part to beginning)
    Returns waveform (np.float32) of shape (N_SAMPLES,)
    """
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True, res_type="kaiser_fast")
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return np.zeros(N_SAMPLES, dtype=np.float32)

    if len(y) == 0:
        return np.zeros(N_SAMPLES, dtype=np.float32)

    # Shift in time domain
    if shift_sec != 0.0:
        shift_samples = int(round(shift_sec * SAMPLE_RATE))
        y = np.roll(y, shift_samples)

    # Fixed length
    if len(y) > N_SAMPLES:
        start = (len(y) - N_SAMPLES) // 2
        y = y[start:start+N_SAMPLES]
    elif len(y) < N_SAMPLES:
        pad_len = N_SAMPLES - len(y)
        y = np.pad(y, (0, pad_len), mode='constant')

    # Standardize per clip
    mean = np.mean(y)
    std = np.std(y)
    if std > 1e-6:
        y = (y - mean) / std
    else:
        y = y - mean
    return y.astype(np.float32)

def extract_mel_deltas(waveform):
    """
    Compute log-mel spectrogram, delta, delta2 and stack.
    waveform: (N_SAMPLES,) at SAMPLE_RATE
    Returns tensor of shape (3, N_MELS, TIME_FRAMES)
    """
    # Mel spectrogram (power)
    mel = librosa.feature.melspectrogram(y=waveform, sr=SAMPLE_RATE, n_mels=N_MELS,
                                         n_fft=N_FFT, hop_length=HOP_LENGTH,
                                         fmin=FMIN, fmax=FMAX, power=2.0)
    # Log scale
    log_mel = librosa.power_to_db(mel, ref=np.max)

    # Deltas
    delta1 = librosa.feature.delta(log_mel, order=1)
    delta2 = librosa.feature.delta(log_mel, order=2)

    # Stack
    features = np.stack([log_mel, delta1, delta2], axis=0)  # (3, N_MELS, T)
    # Per-channel normalization
    for c in range(features.shape[0]):
        mean = np.mean(features[c])
        std = np.std(features[c])
        if std > 1e-6:
            features[c] = (features[c] - mean) / std
        else:
            features[c] = features[c] - mean
    return features.astype(np.float32)

# ----------------------------------------------------------------------
# SpecAugment
# ----------------------------------------------------------------------
class SpecAugment:
    def __init__(self, freq_mask_param=12, time_mask_param=0.2, num_freq_masks=2, num_time_masks=2):
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks

    def __call__(self, spec):
        """
        spec: numpy array shape (C, F, T)
        Returns augmented spec (same shape)
        """
        # Clone to avoid modifying original
        aug = spec.copy()
        C, F, T = aug.shape
        # Frequency masking
        for _ in range(self.num_freq_masks):
            f = np.random.randint(0, self.freq_mask_param)
            f0 = np.random.randint(0, F - f)
            aug[:, f0:f0+f, :] = 0.0
        # Time masking
        max_time = int(self.time_mask_param * T)
        for _ in range(self.num_time_masks):
            t = np.random.randint(1, max_time+1)
            t0 = np.random.randint(0, T - t)
            aug[:, :, t0:t0+t] = 0.0
        return aug

# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------
class MelSpecDataset(Dataset):
    def __init__(self, file_paths, labels=None, augment=False, tta_shifts=None):
        """
        if labels is None -> test mode.
        augment: apply random shift and specaugment during training.
        tta_shifts: list of shifts (seconds) for test-time augmentation.
                    If provided, __getitem__ returns a batch of features for all shifts stacked along first dimension.
        """
        self.file_paths = file_paths
        self.labels = labels
        self.augment = augment
        self.tta_shifts = tta_shifts
        if self.tta_shifts is not None:
            assert labels is None, "TTA only for unlabeled test"
        if augment:
            self.specaug = SpecAugment(freq_mask_param=min(12, N_MELS//3),
                                       time_mask_param=0.2,
                                       num_freq_masks=2,
                                       num_time_masks=2)
        else:
            self.specaug = None

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]

        if self.tta_shifts is None:
            # Single version
            shift_sec = 0.0
            if self.augment:
                # random shift between -0.5 and 0.5 seconds
                shift_sec = np.random.uniform(-0.5, 0.5)
            wave = load_waveform(file_path, shift_sec)
            feat = extract_mel_deltas(wave)
            if self.augment and self.specaug is not None:
                feat = self.specaug(feat)
            feat_tensor = torch.from_numpy(feat)
            if self.labels is not None:
                label = torch.tensor(self.labels[idx], dtype=torch.float32)
                return feat_tensor, label
            else:
                return feat_tensor, os.path.basename(file_path)
        else:
            # TTA: generate features for each shift, stack them along batch dimension later.
            # Instead, we return a list of features? Better to process each shift individually in collate.
            # We'll return a tuple: (list of feature arrays, basename)
            features_list = []
            for shift in self.tta_shifts:
                wave = load_waveform(file_path, shift)
                feat = extract_mel_deltas(wave)
                features_list.append(feat)
            # Convert to tensors
            features_tensors = [torch.from_numpy(f) for f in features_list]
            return features_tensors, os.path.basename(file_path)

def collate_tta(batch):
    """
    Custom collate for TTA dataset: each sample returns a list of tensors and a filename.
    We'll stack them into one big batch with all shifts.
    """
    feats_shifts = []
    names = []
    for item in batch:
        feat_list, name = item
        feats_shifts.extend(feat_list)
        names.append(name)  # name is repeated per shift? But we need to aggregate later.
        # Actually we need to know which predictions belong to which original file.
        # So we will keep the name, and later we will average over shifts per file.
        # So we return a list of names and a batch of features.
    feats_batch = torch.stack(feats_shifts, dim=0)  # (batch*shifts, C, F, T)
    return feats_batch, names  # names length = original batch size

# For standard dataloaders (train/val) we use default collate.
# For test TTA we use custom collate.

# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------
class ResidualBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class Mel2DCNN(nn.Module):
    def __init__(self, input_channels=3, base_channels=32, num_blocks=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.relu = nn.ReLU(inplace=True)

        # Stage 1
        self.stage1 = self._make_layer(base_channels, base_channels, num_blocks, stride=1)
        # Stage 2
        self.stage2 = self._make_layer(base_channels, base_channels*2, num_blocks, stride=2)
        # Stage 3
        self.stage3 = self._make_layer(base_channels*2, base_channels*4, num_blocks, stride=2)

        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(base_channels*4, 1)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(ResidualBlock2D(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock2D(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# ----------------------------------------------------------------------
# Training functions
# ----------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    total_loss = 0.0
    for data, target in loader:
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True).view(-1, 1)
        optimizer.zero_grad()
        with autocast(enabled=USE_AMP):
            output = model(data)
            loss = criterion(output, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * data.size(0)
    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    preds = []
    targets = []
    for data, target in loader:
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True).view(-1, 1)
        output = model(data)
        loss = criterion(output, target)
        total_loss += loss.item() * data.size(0)
        preds.append(torch.sigmoid(output).cpu().numpy())
        targets.append(target.cpu().numpy())
    preds = np.concatenate(preds, axis=0).squeeze()
    targets = np.concatenate(targets, axis=0).squeeze()
    auc = roc_auc_score(targets, preds) if len(np.unique(targets)) > 1 else 0.5
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, auc

def fit(model, train_loader, val_loader, optimizer, criterion, device, epochs, patience, checkpoint_path):
    best_auc = 0.0
    best_epoch = -1
    scaler = GradScaler(enabled=USE_AMP)

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device)
        val_loss, val_auc = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f} - Val AUC: {val_auc:.6f}")

        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch
            torch.save(model.state_dict(), checkpoint_path)

        if epoch - best_epoch >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Load best model
    model.load_state_dict(torch.load(checkpoint_path))
    return best_auc

def predict_test_tta(model, test_dataset, device, batch_size=64):
    """Return dict {filename: probability} averaged over TTA shifts."""
    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True,
                        collate_fn=collate_tta)
    model.eval()
    all_probs = {}
    with torch.no_grad():
        for feats, names in loader:
            feats = feats.to(device, non_blocking=True)
            outputs = model(feats)
            probs = torch.sigmoid(outputs).cpu().numpy().squeeze()
            # Reconstruct per-file average
            num_shifts = len(TTA_SHIFTS)
            for i, name in enumerate(names):
                start = i * num_shifts
                end = (i+1) * num_shifts
                slice_probs = probs[start:end] if num_shifts > 1 else [probs[i]]
                avg_prob = np.mean(slice_probs)
                all_probs[name] = float(avg_prob)
    return all_probs

# ----------------------------------------------------------------------
# Build datasets and dataloaders
# ----------------------------------------------------------------------
# Training dataset with augmentation
train_dataset = MelSpecDataset(train_paths_split, train_labels_split, augment=True)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True, drop_last=False)

# Validation dataset no augmentation
val_dataset = MelSpecDataset(val_paths, val_labels, augment=False)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)

# Compute pos_weight for BCEWithLogitsLoss
pos_count = sum(train_labels_split)
neg_count = len(train_labels_split) - pos_count
pos_weight = torch.tensor([max(1.0, neg_count / pos_count)], device=DEVICE)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# Initialize model
model = Mel2DCNN(input_channels=3, base_channels=32, num_blocks=1).to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# Checkpoint path
ckpt_path = os.path.join(WORKING_DIR, "best_model.pth")

# ----------------------------------------------------------------------
# Train with early stopping
# ----------------------------------------------------------------------
print("Starting training...")
best_val_auc = fit(model, train_loader, val_loader, optimizer, criterion, DEVICE,
                   MAX_EPOCHS, EARLY_STOP_PATIENCE, ckpt_path)
print(f"Best validation AUC: {best_val_auc:.6f}")

# ----------------------------------------------------------------------
# Fine-tune on full training data (train+val)
# ----------------------------------------------------------------------
print("Fine-tuning on full training set...")
full_train_paths = train_paths
full_train_labels = train_labels
full_train_dataset = MelSpecDataset(full_train_paths, full_train_labels, augment=True)
full_train_loader = DataLoader(full_train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=NUM_WORKERS, pin_memory=True)

# Re-initialize optimizer with lower LR
optimizer_ft = optim.AdamW(model.parameters(), lr=FINETUNE_LR, weight_decay=WEIGHT_DECAY)
scaler_ft = GradScaler(enabled=USE_AMP)

for epoch in range(FINETUNE_EPOCHS):
    train_loss = train_one_epoch(model, full_train_loader, optimizer_ft, criterion, scaler_ft, DEVICE)
    print(f"Finetune epoch {epoch+1}/{FINETUNE_EPOCHS} - Train Loss: {train_loss:.6f}")

# ----------------------------------------------------------------------
# Evaluate final model on validation set (no TTA and with TTA)
# ----------------------------------------------------------------------
# No TTA
_, val_auc = evaluate(model, val_loader, criterion, DEVICE)
print(f"Validation AUC (no TTA): {val_auc:.6f}")

# With TTA on validation set (optional)
val_tta_dataset = MelSpecDataset(val_paths, labels=None, augment=False, tta_shifts=TTA_SHIFTS)
val_tta_probs = predict_test_tta(model, val_tta_dataset, DEVICE, batch_size=64)
val_tta_preds = np.array([val_tta_probs[os.path.basename(p)] for p in val_paths])
val_auc_tta = roc_auc_score(val_labels, val_tta_preds)
print(f"Validation AUC (with TTA): {val_auc_tta:.6f}")

# ----------------------------------------------------------------------
# Test set predictions with TTA
# ----------------------------------------------------------------------
test_dataset = MelSpecDataset(test_paths, labels=None, augment=False, tta_shifts=TTA_SHIFTS)
test_probs = predict_test_tta(model, test_dataset, DEVICE, batch_size=64)

# Build submission dataframe
submission = pd.DataFrame({
    "clip": [os.path.basename(p) for p in test_paths],
    "probability": [test_probs[os.path.basename(p)] for p in test_paths]
})
# Ensure sorted as in sample_submission (by clip name)
submission = submission.sort_values("clip")
submission.to_csv(os.path.join(SUBMISSION_DIR, "submission.csv"), index=False)
print(f"Submission saved to {SUBMISSION_DIR}/submission.csv")

print("Script finished.")