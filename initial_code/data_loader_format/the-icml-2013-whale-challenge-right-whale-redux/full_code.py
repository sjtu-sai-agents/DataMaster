import os
import glob
import random
import numpy as np
import pandas as pd
import librosa
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


# Audio preprocessing parameters (defaults, can be overridden via config)
SAMPLE_RATE = 2000
DURATION = 3.0
N_SAMPLES = int(SAMPLE_RATE * DURATION)  # 6000
N_MELS = 64
N_FFT = 512
HOP_LENGTH = 128
FMIN = 30
FMAX = 800


def get_train_files(train_dir):
    """Return list of (file_path, label) for training."""
    files = glob.glob(os.path.join(train_dir, "*.aif"))
    file_label_pairs = []
    for f in files:
        basename = os.path.basename(f)
        if basename.endswith('_1.aif'):
            label = 1
        elif basename.endswith('_0.aif'):
            label = 0
        else:
            parts = basename[:-4].split('_')
            label = int(parts[-1]) if parts[-1].isdigit() else 0
        file_label_pairs.append((f, label))
    return file_label_pairs


def load_waveform(file_path, sample_rate=2000, n_samples=6000, shift_sec=0.0):
    """
    Load audio, resample to sample_rate, fix length to n_samples.
    If shift_sec != 0, circular shift waveform by shift_samples.
    Returns waveform (np.float32) of shape (n_samples,)
    """
    try:
        y, sr = librosa.load(file_path, sr=sample_rate, mono=True, res_type="kaiser_fast")
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return np.zeros(n_samples, dtype=np.float32)

    if len(y) == 0:
        return np.zeros(n_samples, dtype=np.float32)

    # Shift in time domain
    if shift_sec != 0.0:
        shift_samples = int(round(shift_sec * sample_rate))
        y = np.roll(y, shift_samples)

    # Fixed length
    if len(y) > n_samples:
        start = (len(y) - n_samples) // 2
        y = y[start:start + n_samples]
    elif len(y) < n_samples:
        pad_len = n_samples - len(y)
        y = np.pad(y, (0, pad_len), mode='constant')

    # Standardize per clip
    mean = np.mean(y)
    std = np.std(y)
    if std > 1e-6:
        y = (y - mean) / std
    else:
        y = y - mean
    return y.astype(np.float32)


def extract_mel_deltas(waveform, sample_rate=2000, n_mels=64, n_fft=512,
                       hop_length=128, fmin=30, fmax=800):
    """
    Compute log-mel spectrogram, delta, delta2 and stack.
    Returns tensor of shape (3, n_mels, time_frames)
    """
    mel = librosa.feature.melspectrogram(y=waveform, sr=sample_rate, n_mels=n_mels,
                                         n_fft=n_fft, hop_length=hop_length,
                                         fmin=fmin, fmax=fmax, power=2.0)
    log_mel = librosa.power_to_db(mel, ref=np.max)

    delta1 = librosa.feature.delta(log_mel, order=1)
    delta2 = librosa.feature.delta(log_mel, order=2)

    features = np.stack([log_mel, delta1, delta2], axis=0)
    # Per-channel normalization
    for c in range(features.shape[0]):
        mean = np.mean(features[c])
        std = np.std(features[c])
        if std > 1e-6:
            features[c] = (features[c] - mean) / std
        else:
            features[c] = features[c] - mean
    return features.astype(np.float32)


class SpecAugment:
    def __init__(self, freq_mask_param=12, time_mask_param=0.2, num_freq_masks=2, num_time_masks=2):
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks

    def __call__(self, spec):
        """Apply SpecAugment to spectrogram."""
        aug = spec.copy()
        C, F, T = aug.shape
        # Frequency masking
        for _ in range(self.num_freq_masks):
            f = np.random.randint(0, min(self.freq_mask_param, F))
            f0 = np.random.randint(0, max(1, F - f))
            aug[:, f0:f0 + f, :] = 0.0
        # Time masking
        max_time = int(self.time_mask_param * T)
        for _ in range(self.num_time_masks):
            t = np.random.randint(1, max_time + 1)
            t0 = np.random.randint(0, max(1, T - t))
            aug[:, :, t0:t0 + t] = 0.0
        return aug


class MelSpecDataset(Dataset):
    def __init__(self, file_paths, labels=None, augment=False, tta_shifts=None,
                 sample_rate=2000, n_samples=6000, n_mels=64, n_fft=512,
                 hop_length=128, fmin=30, fmax=800):
        self.file_paths = file_paths
        self.labels = labels
        self.augment = augment
        self.tta_shifts = tta_shifts
        self.sample_rate = sample_rate
        self.n_samples = n_samples
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax

        if self.tta_shifts is not None:
            assert labels is None, "TTA only for unlabeled test"
        if augment:
            self.specaug = SpecAugment(freq_mask_param=min(12, n_mels // 3),
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
            shift_sec = 0.0
            if self.augment:
                shift_sec = np.random.uniform(-0.5, 0.5)
            wave = load_waveform(file_path, self.sample_rate, self.n_samples, shift_sec)
            feat = extract_mel_deltas(wave, self.sample_rate, self.n_mels,
                                      self.n_fft, self.hop_length, self.fmin, self.fmax)
            if self.augment and self.specaug is not None:
                feat = self.specaug(feat)
            feat_tensor = torch.from_numpy(feat)
            if self.labels is not None:
                label = torch.tensor(self.labels[idx], dtype=torch.float32)
                return feat_tensor, label
            else:
                return feat_tensor, os.path.basename(file_path)
        else:
            features_list = []
            for shift in self.tta_shifts:
                wave = load_waveform(file_path, self.sample_rate, self.n_samples, shift)
                feat = extract_mel_deltas(wave, self.sample_rate, self.n_mels,
                                         self.n_fft, self.hop_length, self.fmin, self.fmax)
                features_list.append(feat)
            features_tensors = [torch.from_numpy(f) for f in features_list]
            return features_tensors, os.path.basename(file_path)


def collate_tta(batch):
    """Custom collate for TTA dataset."""
    feats_shifts = []
    names = []
    for item in batch:
        feat_list, name = item
        feats_shifts.extend(feat_list)
        names.append(name)
    feats_batch = torch.stack(feats_shifts, dim=0)
    return feats_batch, names


class MyDataLoader(BaseDataLoader):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Extract config parameters with defaults
        self.input_dir = kwargs.get('input_dir', './input')
        self.train_dir = kwargs.get('train_dir', os.path.join(self.input_dir, 'train2'))
        self.test_dir = kwargs.get('test_dir', os.path.join(self.input_dir, 'test2'))
        self.sample_rate = kwargs.get('sample_rate', 2000)
        self.duration = kwargs.get('duration', 3.0)
        self.n_mels = kwargs.get('n_mels', 64)
        self.n_fft = kwargs.get('n_fft', 512)
        self.hop_length = kwargs.get('hop_length', 128)
        self.fmin = kwargs.get('fmin', 30)
        self.fmax = kwargs.get('fmax', 800)
        self.random_seed = kwargs.get('random_seed', 42)
        self.batch_size = kwargs.get('batch_size', 256)
        self.num_workers = kwargs.get('num_workers', min(12, max(1, os.cpu_count() - 1)))
        self.tta_shifts = kwargs.get('tta_shifts', [0.0, 0.25, -0.25, 0.5, -0.5])

        self.n_samples = int(self.sample_rate * self.duration)

    def setup(self):
        """Load data, feature engineering, data augmentation, etc."""
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

        # Get training files
        train_items = get_train_files(self.train_dir)
        if len(train_items) == 0:
            raise RuntimeError("No training files found.")

        train_paths, train_labels = zip(*train_items)
        train_paths, train_labels = list(train_paths), list(train_labels)

        # Get test files
        test_paths = sorted(glob.glob(os.path.join(self.test_dir, "*.aif")))
        if len(test_paths) == 0:
            raise RuntimeError("No test files found.")

        print(f"Found {len(train_paths)} training files, {len(test_paths)} test files.")

        # Handle validation split - use val.csv if exists
        val_csv_path = os.path.join(self.input_dir, 'val.csv')
        if os.path.exists(val_csv_path):
            val_df = pd.read_csv(val_csv_path)
            # Get validation image names
            if 'image' in val_df.columns:
                val_images = set(str(v) for v in val_df['image'].values)
            else:
                val_images = set(str(v) for v in val_df.iloc[:, 0].values)
            # Normalize to basenames
            val_images = set(os.path.basename(v) for v in val_images)

            # Split training data
            train_paths_split = []
            train_labels_split = []
            val_paths = []
            val_labels = []
            for path, label in zip(train_paths, train_labels):
                basename = os.path.basename(path)
                if basename in val_images:
                    val_paths.append(path)
                    val_labels.append(label)
                else:
                    train_paths_split.append(path)
                    train_labels_split.append(label)

            print(f"Using fixed validation set from val.csv: {len(val_paths)} samples")
        else:
            # Fallback to stratified split
            train_idx, val_idx = train_test_split(
                range(len(train_paths)),
                test_size=0.2,
                random_state=self.random_seed,
                stratify=train_labels
            )
            train_paths_split = [train_paths[i] for i in train_idx]
            train_labels_split = [train_labels[i] for i in train_idx]
            val_paths = [train_paths[i] for i in val_idx]
            val_labels = [train_labels[i] for i in val_idx]
            print("Using stratified random split (no val.csv found)")

        print(f"Train split: {len(train_paths_split)} files, positive ratio: {sum(train_labels_split) / len(train_labels_split):.3f}")
        print(f"Val split:   {len(val_paths)} files, positive ratio: {sum(val_labels) / len(val_labels):.3f}")

        # Create datasets
        train_dataset = MelSpecDataset(
            train_paths_split, train_labels_split, augment=True,
            sample_rate=self.sample_rate, n_samples=self.n_samples, n_mels=self.n_mels,
            n_fft=self.n_fft, hop_length=self.hop_length, fmin=self.fmin, fmax=self.fmax
        )

        val_dataset = MelSpecDataset(
            val_paths, val_labels, augment=False,
            sample_rate=self.sample_rate, n_samples=self.n_samples, n_mels=self.n_mels,
            n_fft=self.n_fft, hop_length=self.hop_length, fmin=self.fmin, fmax=self.fmax
        )

        test_dataset = MelSpecDataset(
            test_paths, labels=None, augment=False, tta_shifts=self.tta_shifts,
            sample_rate=self.sample_rate, n_samples=self.n_samples, n_mels=self.n_mels,
            n_fft=self.n_fft, hop_length=self.hop_length, fmin=self.fmin, fmax=self.fmax
        )

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True, drop_last=False
        )

        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True
        )

        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
            collate_fn=collate_tta
        )

        # Store all data needed for training
        self.train_data = {
            'train_loader': train_loader,
            'val_loader': val_loader,
            'train_labels': train_labels_split,
            'val_paths': val_paths,
            'val_labels': val_labels,
            'full_train_paths': train_paths,
            'full_train_labels': train_labels,
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'sample_rate': self.sample_rate,
            'n_samples': self.n_samples,
            'n_mels': self.n_mels,
            'n_fft': self.n_fft,
            'hop_length': self.hop_length,
            'fmin': self.fmin,
            'fmax': self.fmax,
            'tta_shifts': self.tta_shifts,
        }

        self.test_data = {
            'test_loader': test_loader,
            'test_paths': test_paths,
        }

    def describe(self) -> str:
        """Return a description of your data processing approach."""
        return """
        Audio Classification DataLoader:
        - Loads .aif audio files from train and test directories
        - Extracts log-mel spectrograms with delta and delta-delta features (3 channels)
        - Audio parameters: sample_rate=2000, duration=3s, n_mels=64
        - Applies SpecAugment (frequency and time masking) during training
        - Random time shift augmentation during training
        - Test-time augmentation (TTA) with multiple time shifts for inference
        - Uses fixed validation set from val.csv if available, otherwise stratified split
        """

    def get_full_train_loader(self):
        """Get dataloader for full training data (train+val) for fine-tuning."""
        full_train_dataset = MelSpecDataset(
            self.train_data['full_train_paths'],
            self.train_data['full_train_labels'],
            augment=True,
            sample_rate=self.train_data['sample_rate'],
            n_samples=self.train_data['n_samples'],
            n_mels=self.train_data['n_mels'],
            n_fft=self.train_data['n_fft'],
            hop_length=self.train_data['hop_length'],
            fmin=self.train_data['fmin'],
            fmax=self.train_data['fmax']
        )
        return DataLoader(
            full_train_dataset, batch_size=self.train_data['batch_size'],
            shuffle=True, num_workers=self.train_data['num_workers'], pin_memory=True
        )

    def get_val_tta_dataset(self):
        """Get validation dataset with TTA for evaluation."""
        return MelSpecDataset(
            self.train_data['val_paths'], labels=None, augment=False,
            tta_shifts=self.train_data['tta_shifts'],
            sample_rate=self.train_data['sample_rate'],
            n_samples=self.train_data['n_samples'],
            n_mels=self.train_data['n_mels'],
            n_fft=self.train_data['n_fft'],
            hop_length=self.train_data['hop_length'],
            fmin=self.train_data['fmin'],
            fmax=self.train_data['fmax']
        )

import os
import random
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import warnings
warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description='Audio Classification Training Script')
    
    # Path arguments
    parser.add_argument('--input_dir', type=str, default='./input',
                        help='Input directory containing train and test data')
    parser.add_argument('--submission_dir', type=str, default='./submission',
                        help='Directory to save submission files')
    parser.add_argument('--working_dir', type=str, default='./working',
                        help='Working directory for checkpoints')
    
    # Audio parameters
    parser.add_argument('--sample_rate', type=int, default=2000,
                        help='Audio sample rate')
    parser.add_argument('--duration', type=float, default=3.0,
                        help='Audio duration in seconds')
    parser.add_argument('--n_mels', type=int, default=64,
                        help='Number of mel bands')
    parser.add_argument('--n_fft', type=int, default=512,
                        help='FFT window size')
    parser.add_argument('--hop_length', type=int, default=128,
                        help='Hop length for STFT')
    parser.add_argument('--fmin', type=int, default=30,
                        help='Minimum frequency for mel filterbank')
    parser.add_argument('--fmax', type=int, default=800,
                        help='Maximum frequency for mel filterbank')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Number of data loading workers')
    parser.add_argument('--max_epochs', type=int, default=14,
                        help='Maximum number of training epochs')
    parser.add_argument('--early_stop_patience', type=int, default=5,
                        help='Early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for optimizer')
    parser.add_argument('--finetune_epochs', type=int, default=5,
                        help='Number of fine-tuning epochs')
    parser.add_argument('--finetune_lr', type=float, default=5e-4,
                        help='Learning rate for fine-tuning')
    
    # Model parameters
    parser.add_argument('--base_channels', type=int, default=32,
                        help='Base number of channels in model')
    parser.add_argument('--num_blocks', type=int, default=1,
                        help='Number of residual blocks per stage')
    
    # Other parameters
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--use_amp', action='store_true', default=True,
                        help='Use automatic mixed precision')
    parser.add_argument('--no_amp', action='store_false', dest='use_amp',
                        help='Disable automatic mixed precision')
    parser.add_argument('--tta_shifts', type=float, nargs='+', 
                        default=[0.0, 0.25, -0.25, 0.5, -0.5],
                        help='TTA time shifts in seconds')
    
    return parser.parse_args()


# ----------------------------------------------------------------------
# Model Definition
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

        self.stage1 = self._make_layer(base_channels, base_channels, num_blocks, stride=1)
        self.stage2 = self._make_layer(base_channels, base_channels * 2, num_blocks, stride=2)
        self.stage3 = self._make_layer(base_channels * 2, base_channels * 4, num_blocks, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_channels * 4, 1)

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
# Training Functions
# ----------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, criterion, scaler, device, use_amp):
    model.train()
    total_loss = 0.0
    for data, target in loader:
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True).view(-1, 1)
        optimizer.zero_grad()
        with autocast(enabled=use_amp):
            output = model(data)
            loss = criterion(output, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * data.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp):
    model.eval()
    total_loss = 0.0
    preds = []
    targets = []
    for data, target in loader:
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True).view(-1, 1)
        with autocast(enabled=use_amp):
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


def fit(model, train_loader, val_loader, optimizer, criterion, device, epochs, 
        patience, checkpoint_path, use_amp):
    best_auc = 0.0
    best_epoch = -1
    scaler = GradScaler(enabled=use_amp)

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device, use_amp)
        val_loss, val_auc = evaluate(model, val_loader, criterion, device, use_amp)
        print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f} - Val AUC: {val_auc:.6f}")

        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch
            torch.save(model.state_dict(), checkpoint_path)

        if epoch - best_epoch >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    model.load_state_dict(torch.load(checkpoint_path))
    return best_auc


def collate_tta(batch):
    """Custom collate for TTA dataset."""
    feats_shifts = []
    names = []
    for item in batch:
        feat_list, name = item
        feats_shifts.extend(feat_list)
        names.append(name)
    feats_batch = torch.stack(feats_shifts, dim=0)
    return feats_batch, names


def predict_test_tta(model, test_dataset, device, batch_size, num_workers, tta_shifts, use_amp):
    """Return dict {filename: probability} averaged over TTA shifts."""
    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True,
                        collate_fn=collate_tta)
    model.eval()
    all_probs = {}
    with torch.no_grad():
        for feats, names in loader:
            feats = feats.to(device, non_blocking=True)
            with autocast(enabled=use_amp):
                outputs = model(feats)
            probs = torch.sigmoid(outputs).cpu().numpy().squeeze()
            num_shifts = len(tta_shifts)
            for i, name in enumerate(names):
                start = i * num_shifts
                end = (i + 1) * num_shifts
                slice_probs = probs[start:end] if num_shifts > 1 else [probs[i]]
                avg_prob = np.mean(slice_probs)
                all_probs[name] = float(avg_prob)
    return all_probs


def main():
    args = parse_args()
    
    # Set random seeds
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create directories
    os.makedirs(args.submission_dir, exist_ok=True)
    os.makedirs(args.working_dir, exist_ok=True)

    # Number of workers
    num_workers = args.num_workers if args.num_workers else min(12, max(1, os.cpu_count() - 1))

    # Initialize DataLoader
    data_loader = MyDataLoader(
        input_dir=args.input_dir,
        sample_rate=args.sample_rate,
        duration=args.duration,
        n_mels=args.n_mels,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        fmin=args.fmin,
        fmax=args.fmax,
        random_seed=args.random_seed,
        batch_size=args.batch_size,
        num_workers=num_workers,
        tta_shifts=args.tta_shifts,
    )
    
    train_data, test_data = data_loader.get_data()
    
    train_loader = train_data['train_loader']
    val_loader = train_data['val_loader']
    train_labels = train_data['train_labels']
    val_paths = train_data['val_paths']
    val_labels = train_data['val_labels']
    tta_shifts = train_data['tta_shifts']
    
    test_loader = test_data['test_loader']
    test_paths = test_data['test_paths']

    # Compute pos_weight for BCEWithLogitsLoss
    pos_count = sum(train_labels)
    neg_count = len(train_labels) - pos_count
    pos_weight = torch.tensor([max(1.0, neg_count / pos_count)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Initialize model
    model = Mel2DCNN(input_channels=3, base_channels=args.base_channels, 
                     num_blocks=args.num_blocks).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # Checkpoint path
    ckpt_path = os.path.join(args.working_dir, "best_model.pth")

    # Train with early stopping
    print("Starting training...")
    best_val_auc = fit(model, train_loader, val_loader, optimizer, criterion, device,
                       args.max_epochs, args.early_stop_patience, ckpt_path, args.use_amp)
    print(f"Best validation AUC: {best_val_auc:.6f}")

    # Fine-tune on full training data
    print("Fine-tuning on full training set...")
    full_train_loader = data_loader.get_full_train_loader()
    optimizer_ft = optim.AdamW(model.parameters(), lr=args.finetune_lr, weight_decay=args.weight_decay)
    scaler_ft = GradScaler(enabled=args.use_amp)

    for epoch in range(args.finetune_epochs):
        train_loss = train_one_epoch(model, full_train_loader, optimizer_ft, criterion, 
                                      scaler_ft, device, args.use_amp)
        print(f"Finetune epoch {epoch + 1}/{args.finetune_epochs} - Train Loss: {train_loss:.6f}")

    # Evaluate final model on validation set
    _, val_auc = evaluate(model, val_loader, criterion, device, args.use_amp)
    print(f"Validation AUC (no TTA): {val_auc:.6f}")

    # Validation with TTA
    val_tta_dataset = data_loader.get_val_tta_dataset()
    val_tta_probs = predict_test_tta(model, val_tta_dataset, device, args.batch_size, 
                                      num_workers, tta_shifts, args.use_amp)
    val_tta_preds = np.array([val_tta_probs[os.path.basename(p)] for p in val_paths])
    val_auc_tta = roc_auc_score(val_labels, val_tta_preds)
    print(f"Validation AUC (with TTA): {val_auc_tta:.6f}")

    # Test set predictions with TTA
    test_probs = predict_test_tta(model, test_loader.dataset, device, args.batch_size,
                                   num_workers, tta_shifts, args.use_amp)

    # Build submission
    submission = pd.DataFrame({
        "clip": [os.path.basename(p) for p in test_paths],
        "probability": [test_probs[os.path.basename(p)] for p in test_paths]
    })
    submission = submission.sort_values("clip")
    submission.to_csv(os.path.join(args.submission_dir, "submission.csv"), index=False)
    print(f"Submission saved to {args.submission_dir}/submission.csv")

    print("Script finished.")


if __name__ == "__main__":
    main()