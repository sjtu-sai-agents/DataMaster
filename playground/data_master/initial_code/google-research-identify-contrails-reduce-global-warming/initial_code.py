import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Configuration
DATA_DIR = Path("./input")
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "validation"
TEST_DIR = DATA_DIR / "test"
SUBMISSION_DIR = Path("./submission")
WORKING_DIR = Path("./working")
SUBMISSION_DIR.mkdir(exist_ok=True, parents=True)
WORKING_DIR.mkdir(exist_ok=True, parents=True)

BANDS = [f"band_{i:02d}" for i in range(8, 17)]
HEIGHT, WIDTH = 256, 256
TIME_STEPS = 8
BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 1e-3
RANDOM_SEED = 42


# Run-length encoding functions
def rle_encode(x):
    dots = np.where(x.flatten() == 1)[0]
    if len(dots) == 0:
        return "-"
    run_lengths = []
    prev = -2
    for b in dots:
        if b > prev + 1:
            run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return " ".join(str(r) for r in run_lengths)


def rle_decode(mask_rle, shape=(HEIGHT, WIDTH)):
    if mask_rle == "-":
        return np.zeros(shape, dtype=np.uint8)
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        mask[lo:hi] = 1
    return mask.reshape(shape, order="F")


# Dataset class with proper directory handling
class ContrailsDataset2D(Dataset):
    def __init__(self, record_ids, data_dir, is_train=True):
        self.record_ids = record_ids
        self.data_dir = data_dir
        self.is_train = is_train

    def __len__(self):
        return len(self.record_ids)

    def __getitem__(self, idx):
        record_id = self.record_ids[idx]
        record_path = self.data_dir / str(record_id)

        band_data = []
        for band in BANDS:
            band_path = record_path / f"{band}.npy"
            if not band_path.exists():
                raise FileNotFoundError(f"Missing file: {band_path}")
            data = np.load(band_path).astype(np.float32)
            mean = data.mean()
            std = data.std()
            data = (data - mean) / (std + 1e-8)
            band_data.append(data)

        x = np.stack(band_data, axis=-1)
        x = np.transpose(x, (2, 3, 0, 1))
        x = x.reshape(-1, HEIGHT, WIDTH)

        if self.is_train:
            mask_path = record_path / "human_pixel_masks.npy"
            if not mask_path.exists():
                raise FileNotFoundError(f"Missing mask: {mask_path}")
            y = np.load(mask_path).astype(np.float32)
            y = y[:, :, 0]
            return torch.tensor(x), torch.tensor(y)
        else:
            return torch.tensor(x), record_id


# 2D U-Net
class DoubleConv2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv2D(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        self.conv = DoubleConv2D(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet2D(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.inc = DoubleConv2D(n_channels, 64)
        self.down1 = Down2D(64, 128)
        self.down2 = Down2D(128, 256)
        self.down3 = Down2D(256, 512)
        self.up1 = Up2D(512, 256)
        self.up2 = Up2D(256, 128)
        self.up3 = Up2D(128, 64)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits


class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, targets, smooth=1e-6):
        preds = torch.sigmoid(preds)
        preds_flat = preds.view(-1)
        targets_flat = targets.view(-1)
        intersection = (preds_flat * targets_flat).sum()
        dice_loss = 1 - (2.0 * intersection + smooth) / (
            preds_flat.sum() + targets_flat.sum() + smooth
        )
        bce_loss = F.binary_cross_entropy(preds_flat, targets_flat)
        return dice_loss + bce_loss


def dice_score(preds, targets, threshold=0.5):
    preds_bin = (preds > threshold).float()
    targets = targets.float()
    intersection = (preds_bin * targets).sum()
    union = preds_bin.sum() + targets.sum()
    return (2.0 * intersection + 1e-6) / (union + 1e-6)


# Load data correctly
print("Loading data...")
train_ids = [f.name for f in TRAIN_DIR.iterdir() if f.is_dir()]
val_ids = [f.name for f in VAL_DIR.iterdir() if f.is_dir()]
print(f"Train samples: {len(train_ids)}, Val samples: {len(val_ids)}")

train_dataset = ContrailsDataset2D(train_ids, TRAIN_DIR, is_train=True)
val_dataset = ContrailsDataset2D(val_ids, VAL_DIR, is_train=True)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
)

# Model setup
input_channels = len(BANDS) * TIME_STEPS
model = UNet2D(n_channels=input_channels, n_classes=1).to(device)
criterion = DiceBCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", patience=2, factor=0.5
)

# Training
print("Starting training...")
best_val_dice = 0.0
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    train_dice = 0.0

    for batch_x, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y.unsqueeze(1))
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        with torch.no_grad():
            train_dice += dice_score(
                torch.sigmoid(outputs), batch_y.unsqueeze(1)
            ).item()

    model.eval()
    val_loss = 0.0
    val_dice = 0.0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y.unsqueeze(1))
            val_loss += loss.item()
            val_dice += dice_score(torch.sigmoid(outputs), batch_y.unsqueeze(1)).item()

    avg_train_loss = train_loss / len(train_loader)
    avg_train_dice = train_dice / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    avg_val_dice = val_dice / len(val_loader)

    print(f"Epoch {epoch+1}/{EPOCHS}:")
    print(f"Train Loss: {avg_train_loss:.4f}, Train Dice: {avg_train_dice:.4f}")
    print(f"Val Loss: {avg_val_loss:.4f}, Val Dice: {avg_val_dice:.4f}")

    scheduler.step(avg_val_dice)

    if avg_val_dice > best_val_dice:
        best_val_dice = avg_val_dice
        torch.save(model.state_dict(), WORKING_DIR / "best_model.pth")

print(f"Best validation Dice: {best_val_dice:.4f}")

# Load best model
model.load_state_dict(torch.load(WORKING_DIR / "best_model.pth"))

# Test prediction
print("Predicting on test set...")
test_ids = [f.name for f in TEST_DIR.iterdir() if f.is_dir()]
test_dataset = ContrailsDataset2D(test_ids, TEST_DIR, is_train=False)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
)

model.eval()
predictions = []
record_ids = []
with torch.no_grad():
    for batch_x, batch_ids in test_loader:
        batch_x = batch_x.to(device)
        outputs = model(batch_x)
        probs = torch.sigmoid(outputs).cpu().numpy()
        for i in range(probs.shape[0]):
            mask = (probs[i, 0] > 0.5).astype(np.uint8)
            predictions.append(mask)
            record_ids.append(batch_ids[i])

# Create submission
print("Creating submission file...")
submission_data = []
for record_id, mask in zip(record_ids, predictions):
    rle = rle_encode(mask)
    submission_data.append({"record_id": record_id, "encoded_pixels": rle})

submission_df = pd.DataFrame(submission_data)
submission_path = SUBMISSION_DIR / "submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")

# Final validation metric
model.eval()
val_dice_total = 0.0
with torch.no_grad():
    for batch_x, batch_y in val_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        outputs = model(batch_x)
        val_dice_total += dice_score(
            torch.sigmoid(outputs), batch_y.unsqueeze(1)
        ).item()

final_val_dice = val_dice_total / len(val_loader)
print(f"Final Validation Dice Coefficient: {final_val_dice:.4f}")
