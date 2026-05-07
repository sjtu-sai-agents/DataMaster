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