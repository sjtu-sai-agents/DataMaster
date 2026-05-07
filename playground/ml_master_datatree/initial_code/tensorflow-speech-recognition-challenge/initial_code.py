import os
import pandas as pd
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import warnings
import librosa
import random
import math

warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Define paths
BASE_PATH = "./input"
TRAIN_PATH = os.path.join(BASE_PATH, "train", "audio")
TEST_PATH = os.path.join(BASE_PATH, "test", "audio")
SAMPLE_SUB_PATH = os.path.join(BASE_PATH, "sample_submission.csv")
SUBMISSION_DIR = "./submission"
WORKING_DIR = "./working"
os.makedirs(SUBMISSION_DIR, exist_ok=True)
os.makedirs(WORKING_DIR, exist_ok=True)

# Target classes (12 classes)
TARGET_CLASSES = [
    "yes",
    "no",
    "up",
    "down",
    "left",
    "right",
    "on",
    "off",
    "stop",
    "go",
    "silence",
    "unknown",
]
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(TARGET_CLASSES)}
IDX_TO_CLASS = {idx: cls for cls, idx in CLASS_TO_IDX.items()}

# Audio parameters
SAMPLE_RATE = 16000
DURATION = 1.0
N_SAMPLES = int(SAMPLE_RATE * DURATION)
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 64

print("Preparing data...")

# Load training data
train_files = []
train_labels = []

for label_dir in os.listdir(TRAIN_PATH):
    dir_path = os.path.join(TRAIN_PATH, label_dir)
    if not os.path.isdir(dir_path) or label_dir == "_background_noise_":
        continue

    # Map label to target classes
    if label_dir in TARGET_CLASSES[:10]:
        target_label = label_dir
    else:
        target_label = "unknown"

    label_idx = CLASS_TO_IDX[target_label]

    for fname in os.listdir(dir_path):
        if fname.endswith(".wav"):
            full_path = os.path.join(TRAIN_PATH, label_dir, fname)
            train_files.append(full_path)
            train_labels.append(label_idx)

print(f"Total speech command samples: {len(train_files)}")

# Add silence samples from background noise
print("Adding silence samples from background noise...")
noise_dir = os.path.join(TRAIN_PATH, "_background_noise_")
if os.path.exists(noise_dir):
    noise_files = [f for f in os.listdir(noise_dir) if f.endswith(".wav")]
    n_silence_samples = 2000
    silence_idx = CLASS_TO_IDX["silence"]

    for i in range(n_silence_samples):
        noise_file = random.choice(noise_files)
        noise_path = os.path.join(noise_dir, noise_file)
        noise_audio, sr = librosa.load(noise_path, sr=SAMPLE_RATE)

        if len(noise_audio) > N_SAMPLES:
            start = random.randint(0, len(noise_audio) - N_SAMPLES)
            segment = noise_audio[start : start + N_SAMPLES]
        else:
            segment = np.pad(noise_audio, (0, max(0, N_SAMPLES - len(noise_audio))))

        seg_path = os.path.join(WORKING_DIR, f"silence_{i:06d}.npy")
        np.save(seg_path, segment)

        train_files.append(seg_path)
        train_labels.append(silence_idx)

print(f"Total training samples (including silence): {len(train_files)}")

# Split into train and validation sets
train_files, val_files, train_labels, val_labels = train_test_split(
    train_files, train_labels, test_size=0.2, random_state=42, stratify=train_labels
)

print(f"Training samples: {len(train_files)}")
print(f"Validation samples: {len(val_files)}")

# Mel-spectrogram transform
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
)


class SpeechCommandsDataset(Dataset):
    def __init__(self, file_paths, labels, is_train=True):
        self.file_paths = file_paths
        self.labels = labels
        self.is_train = is_train
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=15)
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=10)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]

        if file_path.endswith(".npy"):
            waveform = np.load(file_path)
            waveform = torch.FloatTensor(waveform).unsqueeze(0)
        else:
            waveform, sample_rate = torchaudio.load(file_path)
            if sample_rate != SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(sample_rate, SAMPLE_RATE)
                waveform = resampler(waveform)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if waveform.shape[1] < N_SAMPLES:
            padding = N_SAMPLES - waveform.shape[1]
            waveform = F.pad(waveform, (0, padding))
        elif waveform.shape[1] > N_SAMPLES:
            if self.is_train:
                start = random.randint(0, waveform.shape[1] - N_SAMPLES)
                waveform = waveform[:, start : start + N_SAMPLES]
            else:
                start = (waveform.shape[1] - N_SAMPLES) // 2
                waveform = waveform[:, start : start + N_SAMPLES]

        # Apply data augmentation
        if self.is_train and random.random() > 0.5:
            waveform += torch.randn_like(waveform) * 0.005  # Add small noise

        # Convert to mel-spectrogram
        mel_spec = mel_transform(waveform)
        mel_spec = torchaudio.functional.amplitude_to_DB(
            mel_spec, multiplier=10, amin=1e-10, db_multiplier=0
        )

        # Normalize
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-7)

        # Apply spectrogram augmentation
        if self.is_train:
            if random.random() > 0.5:
                mel_spec = self.time_mask(mel_spec)
            if random.random() > 0.5:
                mel_spec = self.freq_mask(mel_spec)

        return mel_spec, torch.tensor(label, dtype=torch.long)


# SE Block
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# ResNet Block with SE
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class AudioResNet(nn.Module):
    def __init__(self, num_classes=12):
        super(AudioResNet, self).__init__()

        # Initial convolution
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        # ResNet layers
        self.layer1 = self._make_layer(32, 64, 2, stride=2)  # 64x51
        self.layer2 = self._make_layer(64, 128, 2, stride=2)  # 128x26
        self.layer3 = self._make_layer(128, 256, 2, stride=2)  # 256x13

        # Global pooling and FC
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(ResBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


# Create datasets and dataloaders
train_dataset = SpeechCommandsDataset(train_files, train_labels, is_train=True)
val_dataset = SpeechCommandsDataset(val_files, val_labels, is_train=False)

train_loader = DataLoader(
    train_dataset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=64, shuffle=False, num_workers=8, pin_memory=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = AudioResNet(len(TARGET_CLASSES)).to(device)
model_path = os.path.join(WORKING_DIR, "best_model_resnet.pth")

# Check if we have a compatible saved model
train_new_model = True
if os.path.exists(model_path):
    try:
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)
        print("Loaded existing model from checkpoint.")
        train_new_model = False
    except Exception as e:
        print(f"Error loading model: {e}. Training new model.")
        train_new_model = True
else:
    print("No existing model found. Training new model.")

# Train model if needed
if train_new_model:
    print("Training new model...")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=20, eta_min=1e-6
    )

    num_epochs = 25
    best_val_acc = 0
    patience = 8
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for mel_specs, labels in pbar:
            mel_specs, labels = mel_specs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(mel_specs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            pbar.set_postfix(
                {
                    "loss": train_loss / (pbar.n + 1),
                    "acc": 100 * train_correct / train_total,
                }
            )

        train_acc = 100 * train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            for mel_specs, labels in pbar:
                mel_specs, labels = mel_specs.to(device), labels.to(device)

                outputs = model(mel_specs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                pbar.set_postfix(
                    {
                        "loss": val_loss / (pbar.n + 1),
                        "acc": 100 * val_correct / val_total,
                    }
                )

        val_acc = 100 * val_correct / val_total
        print(f"Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
            print(f"Saved new best model with val acc: {val_acc:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    final_val_acc = best_val_acc
else:
    # Calculate validation accuracy for existing model
    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for mel_specs, labels in val_loader:
            mel_specs, labels = mel_specs.to(device), labels.to(device)
            outputs = model(mel_specs)
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    final_val_acc = 100 * val_correct / val_total
    print(f"Validation accuracy of loaded model: {final_val_acc:.2f}%")

# Load best model for testing
model.load_state_dict(torch.load(model_path))
model.eval()

# Prepare test data
test_files = []
test_fnames = []
for fname in sorted(os.listdir(TEST_PATH)):
    if fname.endswith(".wav"):
        test_files.append(os.path.join(TEST_PATH, fname))
        test_fnames.append(fname)

print(f"Number of test files: {len(test_files)}")


class TestDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        waveform, sample_rate = torchaudio.load(file_path)

        if sample_rate != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sample_rate, SAMPLE_RATE)
            waveform = resampler(waveform)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if waveform.shape[1] < N_SAMPLES:
            padding = N_SAMPLES - waveform.shape[1]
            waveform = F.pad(waveform, (0, padding))
        elif waveform.shape[1] > N_SAMPLES:
            start = (waveform.shape[1] - N_SAMPLES) // 2
            waveform = waveform[:, start : start + N_SAMPLES]

        # Convert to mel-spectrogram
        mel_spec = mel_transform(waveform)
        mel_spec = torchaudio.functional.amplitude_to_DB(
            mel_spec, multiplier=10, amin=1e-10, db_multiplier=0
        )
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-7)

        return mel_spec


test_dataset = TestDataset(test_files)
test_loader = DataLoader(
    test_dataset, batch_size=128, shuffle=False, num_workers=8, pin_memory=True
)

# Generate predictions
predictions = []
model.eval()
with torch.no_grad():
    pbar = tqdm(test_loader, desc="Generating predictions")
    for batch in pbar:
        batch = batch.to(device)
        outputs = model(batch)
        _, batch_preds = torch.max(outputs, 1)
        predictions.extend(batch_preds.cpu().numpy())

# Map indices back to labels
pred_labels = [IDX_TO_CLASS[pred] for pred in predictions]

# Create submission file
submission = pd.DataFrame({"fname": test_fnames, "label": pred_labels})

assert len(submission) == len(
    test_files
), f"Expected {len(test_files)} predictions, got {len(submission)}"

# Save submission
submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
submission.to_csv(submission_path, index=False)

print(f"Submission saved to {submission_path}")
print(f"Submission shape: {submission.shape}")
print(f"\nFinal validation accuracy: {final_val_acc:.2f}%")
print("\nPrediction class distribution:")
print(submission["label"].value_counts())
print("\nFirst few predictions:")
print(submission.head())

# Validation metric
print(f"\nValidation accuracy on hold-out set: {final_val_acc:.2f}%")
