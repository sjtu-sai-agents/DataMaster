import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
from tqdm import tqdm
import warnings
from sklearn.model_selection import KFold
import math

warnings.filterwarnings("ignore")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
PATCH_SIZE = 256
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 1e-4
NUM_SLICES = 65
THRESHOLD = 0.5
N_FOLDS = 3

# Paths
INPUT_DIR = "./input"
TRAIN_DIR = os.path.join(INPUT_DIR, "train")
TEST_DIR = os.path.join(INPUT_DIR, "test")
SUBMISSION_DIR = "./submission"
WORKING_DIR = "./working"
os.makedirs(SUBMISSION_DIR, exist_ok=True)
os.makedirs(WORKING_DIR, exist_ok=True)

# Load training fragments
train_fragments = ["1", "2"]
test_fragments = [
    f for f in os.listdir(TEST_DIR) if os.path.isdir(os.path.join(TEST_DIR, f))
]
print(f"Train fragments: {train_fragments}")
print(f"Test fragments: {test_fragments}")


def load_mask(fragment_id, is_train=True):
    path = os.path.join(TRAIN_DIR if is_train else TEST_DIR, fragment_id, "mask.png")
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    mask = (mask > 0).astype(np.uint8)
    return mask


def load_inklabels(fragment_id):
    path = os.path.join(TRAIN_DIR, fragment_id, "inklabels.png")
    ink = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    ink = (ink > 0).astype(np.uint8)
    return ink


def load_volume(fragment_id, is_train=True, slice_range=None):
    base_path = os.path.join(
        TRAIN_DIR if is_train else TEST_DIR, fragment_id, "surface_volume"
    )
    slice_files = sorted(glob.glob(os.path.join(base_path, "*.tif")))
    if slice_range is not None:
        slice_files = slice_files[slice_range[0] : slice_range[1]]
    slices = []
    for f in slice_files[:NUM_SLICES]:  # Ensure we only load NUM_SLICES
        img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        slices.append(img)
    volume = np.stack(slices, axis=0)  # Shape: (depth, height, width)
    return volume.astype(np.float32)


def normalize_volume(volume):
    normalized = np.zeros_like(volume, dtype=np.float32)
    for i in range(volume.shape[0]):
        slice_img = volume[i]
        if np.std(slice_img) > 0:
            normalized[i] = (slice_img - np.mean(slice_img)) / np.std(slice_img)
        else:
            normalized[i] = slice_img - np.mean(slice_img)
    return normalized


class InkDataset(Dataset):
    def __init__(self, volume, mask, ink=None, patch_size=PATCH_SIZE, is_train=True):
        self.volume = volume  # (depth, H, W)
        self.mask = mask  # (H, W)
        self.ink = ink  # (H, W) or None
        self.patch_size = patch_size
        self.is_train = is_train

        # Get valid coordinates (within mask)
        self.coords = []
        H, W = mask.shape
        step = patch_size // 2
        for y in range(0, H - patch_size + 1, step):
            for x in range(0, W - patch_size + 1, step):
                patch_mask = mask[y : y + patch_size, x : x + patch_size]
                if (
                    np.sum(patch_mask) > 0.5 * patch_size * patch_size
                ):  # At least 50% valid
                    self.coords.append((y, x))

        if is_train and len(self.coords) > 5000:
            # Subsample for training efficiency
            indices = np.random.choice(len(self.coords), 5000, replace=False)
            self.coords = [self.coords[i] for i in indices]

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        y, x = self.coords[idx]
        patch_volume = self.volume[
            :, y : y + self.patch_size, x : x + self.patch_size
        ]  # (depth, patch, patch)
        patch_mask = self.mask[y : y + self.patch_size, x : x + self.patch_size]

        # Apply mask and normalize
        patch_volume = patch_volume * patch_mask

        # Additional augmentation for training
        if self.is_train and self.ink is not None:
            patch_ink = self.ink[y : y + self.patch_size, x : x + self.patch_size]
            # Random flip
            if np.random.random() > 0.5:
                patch_volume = np.flip(patch_volume, axis=2).copy()
                patch_ink = np.flip(patch_ink, axis=1).copy()
            if np.random.random() > 0.5:
                patch_volume = np.flip(patch_volume, axis=1).copy()
                patch_ink = np.flip(patch_ink, axis=0).copy()
            # Random rotation
            k = np.random.randint(0, 4)
            patch_volume = np.rot90(patch_volume, k=k, axes=(1, 2)).copy()
            patch_ink = np.rot90(patch_ink, k=k).copy()

            return torch.FloatTensor(patch_volume), torch.FloatTensor(
                patch_ink[None, ...]
            )
        elif self.ink is not None:
            patch_ink = self.ink[y : y + self.patch_size, x : x + self.patch_size]
            return torch.FloatTensor(patch_volume), torch.FloatTensor(
                patch_ink[None, ...]
            )
        else:
            return torch.FloatTensor(patch_volume), torch.FloatTensor(
                patch_mask[None, ...]
            )


class UNet(nn.Module):
    def __init__(self, in_channels=65, out_channels=1):
        super().__init__()
        self.inc = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.outc = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)

        x = self.up1(x3)
        # Handle size mismatch
        diffY = x2.size()[2] - x.size()[2]
        diffX = x2.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x, x2], dim=1)
        x = self.conv1(x)

        x = self.up2(x)
        diffY = x1.size()[2] - x.size()[2]
        diffX = x1.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x, x1], dim=1)
        x = self.conv2(x)

        return torch.sigmoid(self.outc(x))


def dice_loss(pred, target, smooth=1e-7):
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    intersection = (pred * target).sum()
    return 1 - (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def f_beta_score(pred, target, beta=0.5, eps=1e-7):
    pred = (pred > THRESHOLD).float()
    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    fn = ((1 - pred) * target).sum()
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f_beta = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall + eps)
    return f_beta.item()


# Load all training data
print("Loading training data...")
volumes = {}
masks = {}
inks = {}

for frag in train_fragments:
    volumes[frag] = load_volume(frag, is_train=True)
    volumes[frag] = normalize_volume(volumes[frag])
    masks[frag] = load_mask(frag, is_train=True)
    inks[frag] = load_inklabels(frag)
    print(
        f"Loaded fragment {frag}: volume shape {volumes[frag].shape}, mask shape {masks[frag].shape}"
    )

# Prepare cross-validation
print("\nPreparing cross-validation...")
all_coords = []
all_fragments = []
for frag in train_fragments:
    H, W = masks[frag].shape
    step = PATCH_SIZE // 2
    for y in range(0, H - PATCH_SIZE + 1, step):
        for x in range(0, W - PATCH_SIZE + 1, step):
            patch_mask = masks[frag][y : y + PATCH_SIZE, x : x + PATCH_SIZE]
            if np.sum(patch_mask) > 0.5 * PATCH_SIZE * PATCH_SIZE:
                all_coords.append((frag, y, x))
                all_fragments.append(frag)

# Create KFold split
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
fold_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(all_coords)):
    print(f"\n=== Training Fold {fold + 1}/{N_FOLDS} ===")

    # Create datasets
    train_coords = [all_coords[i] for i in train_idx]
    val_coords = [all_coords[i] for i in val_idx]

    class FoldDataset(Dataset):
        def __init__(self, coords, is_train=True):
            self.coords = coords
            self.is_train = is_train

        def __len__(self):
            return len(self.coords)

        def __getitem__(self, idx):
            frag, y, x = self.coords[idx]
            patch_volume = volumes[frag][:, y : y + PATCH_SIZE, x : x + PATCH_SIZE]
            patch_mask = masks[frag][y : y + PATCH_SIZE, x : x + PATCH_SIZE]
            patch_ink = inks[frag][y : y + PATCH_SIZE, x : x + PATCH_SIZE]

            patch_volume = patch_volume * patch_mask

            if self.is_train:
                if np.random.random() > 0.5:
                    patch_volume = np.flip(patch_volume, axis=2).copy()
                    patch_ink = np.flip(patch_ink, axis=1).copy()
                if np.random.random() > 0.5:
                    patch_volume = np.flip(patch_volume, axis=1).copy()
                    patch_ink = np.flip(patch_ink, axis=0).copy()

            return torch.FloatTensor(patch_volume), torch.FloatTensor(
                patch_ink[None, ...]
            )

    train_dataset = FoldDataset(train_coords, is_train=True)
    val_dataset = FoldDataset(val_coords, is_train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    # Initialize model
    model = UNet(in_channels=NUM_SLICES, out_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    # Training loop
    best_f05 = 0.0
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            volume, ink = batch
            volume, ink = volume.to(device), ink.to(device)
            optimizer.zero_grad()
            pred = model(volume)
            loss = dice_loss(pred, ink)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_f05 = 0.0
        with torch.no_grad():
            for volume, ink in val_loader:
                volume, ink = volume.to(device), ink.to(device)
                pred = model(volume)
                val_f05 += f_beta_score(pred, ink) * volume.size(0)

        val_f05 /= len(val_dataset)
        scheduler.step(val_f05)

        print(
            f"Fold {fold+1}, Epoch {epoch+1}: Train Loss = {train_loss/len(train_loader):.4f}, Val F0.5 = {val_f05:.4f}"
        )

        if val_f05 > best_f05:
            best_f05 = val_f05
            torch.save(
                model.state_dict(),
                os.path.join(WORKING_DIR, f"best_model_fold{fold}.pth"),
            )

    fold_scores.append(best_f05)
    print(f"Fold {fold+1} best F0.5: {best_f05:.4f}")

avg_score = np.mean(fold_scores)
print(f"\nAverage cross-validation F0.5 score: {avg_score:.4f}")

# Train final model on all data
print("\n=== Training final model on all data ===")


class FullDataset(Dataset):
    def __init__(self, is_train=True):
        self.coords = all_coords
        self.is_train = is_train

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        frag, y, x = self.coords[idx]
        patch_volume = volumes[frag][:, y : y + PATCH_SIZE, x : x + PATCH_SIZE]
        patch_mask = masks[frag][y : y + PATCH_SIZE, x : x + PATCH_SIZE]
        patch_ink = inks[frag][y : y + PATCH_SIZE, x : x + PATCH_SIZE]

        patch_volume = patch_volume * patch_mask

        if self.is_train:
            if np.random.random() > 0.5:
                patch_volume = np.flip(patch_volume, axis=2).copy()
                patch_ink = np.flip(patch_ink, axis=1).copy()
            if np.random.random() > 0.5:
                patch_volume = np.flip(patch_volume, axis=1).copy()
                patch_ink = np.flip(patch_ink, axis=0).copy()

        return torch.FloatTensor(patch_volume), torch.FloatTensor(patch_ink[None, ...])


full_dataset = FullDataset(is_train=True)
full_loader = DataLoader(
    full_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True
)

final_model = UNet(in_channels=NUM_SLICES, out_channels=1).to(device)
optimizer = torch.optim.Adam(final_model.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCHS):
    final_model.train()
    train_loss = 0.0
    for batch in tqdm(full_loader, desc=f"Final Epoch {epoch+1}/{EPOCHS}"):
        volume, ink = batch
        volume, ink = volume.to(device), ink.to(device)
        optimizer.zero_grad()
        pred = final_model(volume)
        loss = dice_loss(pred, ink)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    print(f"Final Epoch {epoch+1}: Loss = {train_loss/len(full_loader):.4f}")

torch.save(final_model.state_dict(), os.path.join(WORKING_DIR, "final_model.pth"))


# RLE encoding function
def rle_encode(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


# Predict on test fragments
print("\n=== Making predictions on test data ===")
final_model.eval()
submission_data = []

for fragment_id in test_fragments:
    print(f"Processing test fragment {fragment_id}...")
    volume = load_volume(fragment_id, is_train=False)
    volume = normalize_volume(volume)
    mask = load_mask(fragment_id, is_train=False)

    H, W = mask.shape
    pred_full = np.zeros((H, W), dtype=np.float32)
    count = np.zeros((H, W), dtype=np.float32)

    # Create patches with overlap
    step = PATCH_SIZE // 2
    coords = []
    for y in range(0, H - PATCH_SIZE + 1, step):
        for x in range(0, W - PATCH_SIZE + 1, step):
            patch_mask = mask[y : y + PATCH_SIZE, x : x + PATCH_SIZE]
            if np.sum(patch_mask) > 0.1 * PATCH_SIZE * PATCH_SIZE:
                coords.append((y, x))

    print(f"Processing {len(coords)} patches...")
    for y, x in tqdm(coords):
        patch_volume = volume[:, y : y + PATCH_SIZE, x : x + PATCH_SIZE]
        patch_mask = mask[y : y + PATCH_SIZE, x : x + PATCH_SIZE]
        patch_volume = patch_volume * patch_mask
        patch_tensor = torch.FloatTensor(patch_volume).unsqueeze(0).to(device)

        with torch.no_grad():
            patch_pred = final_model(patch_tensor).squeeze().cpu().numpy()

        pred_full[y : y + PATCH_SIZE, x : x + PATCH_SIZE] += patch_pred
        count[y : y + PATCH_SIZE, x : x + PATCH_SIZE] += 1

    # Average overlapping regions
    pred_full = np.divide(
        pred_full, count, out=np.zeros_like(pred_full), where=count != 0
    )
    pred_binary = (pred_full > THRESHOLD).astype(np.uint8) * mask

    # RLE encode
    rle = rle_encode(pred_binary)
    submission_data.append({"Id": fragment_id, "Predicted": rle})

# Create submission file
submission_df = pd.DataFrame(submission_data, columns=["Id", "Predicted"])
submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
submission_df.to_csv(submission_path, index=False)
print(f"\nSubmission saved to {submission_path}")
print(f"Average cross-validation F0.5 score: {avg_score:.4f}")

# Also save to working directory for backup
backup_path = os.path.join(WORKING_DIR, "submission.csv")
submission_df.to_csv(backup_path, index=False)
print(f"Backup saved to {backup_path}")

print("\n=== Submission Format Check ===")
print(f"Number of test fragments: {len(test_fragments)}")
print(f"Submission shape: {submission_df.shape}")
print("Submission head:")
print(submission_df.head())
