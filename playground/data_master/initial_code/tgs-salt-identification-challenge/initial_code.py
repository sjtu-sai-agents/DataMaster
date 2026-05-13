import os
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
import cv2
from tqdm import tqdm
import warnings
import albumentations as A

warnings.filterwarnings("ignore")

# Set paths
BASE_PATH = Path("./input")
TRAIN_IMG_PATH = BASE_PATH / "train" / "images"
TRAIN_MASK_PATH = BASE_PATH / "train" / "masks"
TEST_IMG_PATH = BASE_PATH / "test" / "images"
WORKING_PATH = Path("./working")
SUBMISSION_PATH = Path("./submission")
SUBMISSION_PATH.mkdir(exist_ok=True)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load data
depths_df = pd.read_csv(BASE_PATH / "depths.csv")
train_df = pd.read_csv(BASE_PATH / "train.csv")
sample_sub = pd.read_csv(BASE_PATH / "sample_submission.csv")

# Prepare training data
train_df = train_df.merge(depths_df, on="id")
train_df["image_path"] = train_df["id"].apply(lambda x: TRAIN_IMG_PATH / f"{x}.png")
train_df["mask_path"] = train_df["id"].apply(lambda x: TRAIN_MASK_PATH / f"{x}.png")
train_df["mask_exists"] = train_df["rle_mask"].notna()

# Split train/validation
train_ids, val_ids = train_test_split(
    train_df["id"].values,
    test_size=0.2,
    random_state=42,
    stratify=train_df["mask_exists"].values,
)
train_data = train_df[train_df["id"].isin(train_ids)].copy()
val_data = train_df[train_df["id"].isin(val_ids)].copy()

print(f"Training samples: {len(train_data)}, Validation samples: {len(val_data)}")

# Prepare test data
test_ids = [f.stem for f in TEST_IMG_PATH.glob("*.png")]
test_df = pd.DataFrame({"id": test_ids})
test_df = test_df.merge(depths_df, on="id", how="left")
test_df["image_path"] = test_df["id"].apply(lambda x: TEST_IMG_PATH / f"{x}.png")


# Dataset class
class SaltDataset(Dataset):
    def __init__(self, df, transform=None, training=True):
        self.df = df
        self.transform = transform
        self.training = training

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["image_path"]

        # Load image
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        image = image.astype(np.float32) / 255.0

        # Load depth and normalize
        depth = np.array([row["z"]], dtype=np.float32)
        depth_norm = (depth - 500) / 500  # Normalize around mean depth

        if self.training:
            # Load mask
            if pd.isna(row["rle_mask"]):
                mask = np.zeros((101, 101), dtype=np.float32)
            else:
                mask = cv2.imread(str(row["mask_path"]), cv2.IMREAD_GRAYSCALE)
                mask = mask.astype(np.float32) / 255.0
        else:
            mask = np.zeros((101, 101), dtype=np.float32)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # Stack image with depth channel
        image = np.stack([image, image, image], axis=0)  # Convert to 3-channel
        image = torch.FloatTensor(image)
        mask = torch.FloatTensor(mask).unsqueeze(0)

        return {
            "image": image,
            "mask": mask,
            "depth": torch.FloatTensor(depth_norm),
            "id": row["id"],
        }


# Simple augmentations
train_transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
    ]
)
val_transform = None

# Create datasets and dataloaders
train_dataset = SaltDataset(train_data, transform=train_transform, training=True)
val_dataset = SaltDataset(val_data, transform=val_transform, training=True)
test_dataset = SaltDataset(test_df, transform=None, training=False)

train_loader = DataLoader(
    train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True
)
test_loader = DataLoader(
    test_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True
)


# Fixed model definition with proper dimension handling
class DepthConditionedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.depth_scale = nn.Linear(1, out_channels)
        self.depth_shift = nn.Linear(1, out_channels)

    def forward(self, x, depth):
        x = self.conv(x)
        depth = depth.unsqueeze(-1)
        scale = self.depth_scale(depth).view(x.size(0), -1, 1, 1)
        shift = self.depth_shift(depth).view(x.size(0), -1, 1, 1)
        return x * (1 + scale) + shift


class SimpleSegFormer(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder with padding to maintain dimensions
        self.enc1 = DepthConditionedConv(3, 32)
        self.enc2 = DepthConditionedConv(32, 64)
        self.enc3 = DepthConditionedConv(64, 128)
        self.enc4 = DepthConditionedConv(128, 256)

        # Use stride 2 convolutions instead of maxpool for better control
        self.down1 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.down2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.down3 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)

        # Decoder with bilinear upsampling to handle odd dimensions
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec1 = DepthConditionedConv(256 + 128, 128)

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec2 = DepthConditionedConv(128 + 64, 64)

        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec3 = DepthConditionedConv(64 + 32, 32)

        self.up4 = nn.Upsample(size=(101, 101), mode="bilinear", align_corners=True)
        self.final = nn.Conv2d(32, 1, kernel_size=1)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)

    def forward(self, x, depth):
        # Encoder path
        e1 = F.relu(self.bn1(self.enc1(x, depth)))  # 101x101
        p1 = self.down1(e1)  # 51x51

        e2 = F.relu(self.bn2(self.enc2(p1, depth)))  # 51x51
        p2 = self.down2(e2)  # 26x26

        e3 = F.relu(self.bn3(self.enc3(p2, depth)))  # 26x26
        p3 = self.down3(e3)  # 13x13

        e4 = F.relu(self.bn4(self.enc4(p3, depth)))  # 13x13

        # Decoder path with skip connections
        d1 = self.up1(e4)  # 26x26
        # Pad if necessary to match e3 dimensions
        if d1.shape[-1] != e3.shape[-1] or d1.shape[-2] != e3.shape[-2]:
            diff_h = e3.shape[-2] - d1.shape[-2]
            diff_w = e3.shape[-1] - d1.shape[-1]
            d1 = F.pad(d1, (0, diff_w, 0, diff_h))
        d1 = torch.cat([d1, e3], dim=1)
        d1 = F.relu(self.dec1(d1, depth))

        d2 = self.up2(d1)  # 52x52
        # Pad if necessary to match e2 dimensions
        if d2.shape[-1] != e2.shape[-1] or d2.shape[-2] != e2.shape[-2]:
            diff_h = e2.shape[-2] - d2.shape[-2]
            diff_w = e2.shape[-1] - d2.shape[-1]
            d2 = F.pad(d2, (0, diff_w, 0, diff_h))
        d2 = torch.cat([d2, e2], dim=1)
        d2 = F.relu(self.dec2(d2, depth))

        d3 = self.up3(d2)  # 104x104
        # Crop if necessary to match e1 dimensions
        if d3.shape[-1] != e1.shape[-1] or d3.shape[-2] != e1.shape[-2]:
            d3 = d3[:, :, : e1.shape[-2], : e1.shape[-1]]
        d3 = torch.cat([d3, e1], dim=1)
        d3 = F.relu(self.dec3(d3, depth))

        out = self.up4(d3)  # 101x101
        out = self.final(out)
        # Remove sigmoid activation here - will be handled by BCEWithLogitsLoss
        return out


# Loss functions
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # Apply sigmoid to get probabilities since model outputs logits
        pred = torch.sigmoid(pred)
        pred = pred.view(-1)
        target = target.view(-1)

        intersection = (pred * target).sum()
        dice = (2.0 * intersection + self.smooth) / (
            pred.sum() + target.sum() + self.smooth
        )
        return 1 - dice


class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.7):
        super().__init__()
        self.dice = DiceLoss()
        # Use BCEWithLogitsLoss instead of BCELoss for autocast safety
        self.bce = nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, target):
        dice_loss = self.dice(pred, target)
        bce_loss = self.bce(pred, target)
        return self.alpha * dice_loss + (1 - self.alpha) * bce_loss


# Evaluation metric - mAP at IoU thresholds
def compute_iou(pred_mask, true_mask):
    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()
    return intersection / (union + 1e-6)


def compute_map(predictions, targets, thresholds=np.arange(0.5, 1.0, 0.05)):
    aps = []

    for pred, target in zip(predictions, targets):
        # Apply sigmoid to get probabilities from logits
        pred_prob = 1 / (1 + np.exp(-pred))
        pred_binary = (pred_prob > 0.5).astype(np.float32)
        target_binary = (target > 0.5).astype(np.float32)

        # If both are empty, IoU = 1
        if pred_binary.sum() == 0 and target_binary.sum() == 0:
            iou = 1.0
        else:
            iou = compute_iou(pred_binary, target_binary)

        # Compute precision at each threshold
        precisions = []
        for thresh in thresholds:
            if iou > thresh:
                precisions.append(1.0)
            else:
                precisions.append(0.0)

        aps.append(np.mean(precisions))

    return np.mean(aps)


# Training setup
model = SimpleSegFormer().to(device)
criterion = CombinedLoss(alpha=0.7)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", patience=3, factor=0.5
)
scaler = GradScaler()

# Training loop
best_map = 0.0
epochs = 10  # Reduced for faster execution

for epoch in range(epochs):
    # Training
    model.train()
    train_loss = 0.0
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")

    for batch in train_bar:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        depths = batch["depth"].to(device)

        optimizer.zero_grad()

        with autocast():
            outputs = model(images, depths)
            loss = criterion(outputs, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        train_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    # Validation
    model.eval()
    val_predictions = []
    val_targets = []
    val_loss = 0.0

    with torch.no_grad():
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
        for batch in val_bar:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            depths = batch["depth"].to(device)

            with autocast():
                outputs = model(images, depths)
                loss = criterion(outputs, masks)

            val_loss += loss.item()

            # Store predictions and targets for mAP computation
            outputs_np = outputs.cpu().numpy()
            masks_np = masks.cpu().numpy()

            for i in range(outputs_np.shape[0]):
                val_predictions.append(outputs_np[i, 0])
                val_targets.append(masks_np[i, 0])

    # Compute mAP
    val_map = compute_map(val_predictions, val_targets)
    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)

    print(
        f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val mAP: {val_map:.4f}"
    )

    # Save best model
    if val_map > best_map:
        best_map = val_map
        torch.save(model.state_dict(), WORKING_PATH / "best_model.pth")
        print(f"Best model saved with mAP: {best_map:.4f}")

    scheduler.step(val_map)

    # Early stopping
    if epoch > 5 and val_map < 0.01:
        print("Early stopping triggered")
        break

print(f"Best validation mAP: {best_map:.4f}")

# Load best model for testing
model.load_state_dict(torch.load(WORKING_PATH / "best_model.pth"))
model.eval()


# Test prediction
def rle_encode(img):
    """
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formatted
    """
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


test_predictions = []
test_ids = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Predicting on test set"):
        images = batch["image"].to(device)
        depths = batch["depth"].to(device)
        ids = batch["id"]

        outputs = model(images, depths)
        # Apply sigmoid to get probabilities from logits
        outputs_sigmoid = torch.sigmoid(outputs)
        outputs_np = outputs_sigmoid.cpu().numpy()

        for i in range(outputs_np.shape[0]):
            pred_mask = (outputs_np[i, 0] > 0.5).astype(np.uint8)
            rle = rle_encode(pred_mask)
            test_predictions.append(rle)
            test_ids.append(ids[i])

# Create submission file
submission_df = pd.DataFrame({"id": test_ids, "rle_mask": test_predictions})

# Ensure all test IDs are included
all_test_ids = [f.stem for f in TEST_IMG_PATH.glob("*.png")]
missing_ids = set(all_test_ids) - set(test_ids)
if missing_ids:
    print(f"Warning: {len(missing_ids)} test IDs missing from predictions")
    # Add empty predictions for missing IDs
    for missing_id in missing_ids:
        submission_df = pd.concat(
            [submission_df, pd.DataFrame({"id": [missing_id], "rle_mask": ["1 1"]})],
            ignore_index=True,
        )

# Sort by ID to match expected order
submission_df = submission_df.sort_values("id")

# Save submission
submission_file = SUBMISSION_PATH / "submission.csv"
submission_df.to_csv(submission_file, index=False)
print(f"Submission saved to {submission_file}")
print(f"Submission shape: {submission_df.shape}")
print(f"Sample submission:\n{submission_df.head()}")

# Validate the submission
if submission_file.exists():
    print("\nSubmission validation:")
    print(f"- File exists: Yes")
    print(f"- File size: {os.path.getsize(submission_file)} bytes")
    print(f"- Number of rows: {len(submission_df)}")
    print(f"- Expected rows: {len(all_test_ids)}")
    print(f"- Columns: {list(submission_df.columns)}")

    # Check RLE format
    valid_rle = True
    for rle in submission_df["rle_mask"].head(10):
        parts = str(rle).split()
        if len(parts) % 2 != 0:
            valid_rle = False
            break
        try:
            [int(x) for x in parts]
        except:
            valid_rle = False
            break

    print(f"- Valid RLE format: {'Yes' if valid_rle else 'No'}")

    if len(submission_df) == len(all_test_ids) and valid_rle:
        print("✓ Submission appears to be in correct format.")
    else:
        print("⚠ Submission may have issues. Please check the format.")
else:
    print("✗ Submission file not created!")

print(f"\nFinal validation mAP: {best_map:.4f}")
