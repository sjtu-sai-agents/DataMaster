import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold
import cv2
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import warnings
from collections import defaultdict
import gc

warnings.filterwarnings("ignore")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Constants
IMG_SIZE = 256
BATCH_SIZE = 16
EPOCHS = 15
LR = 1e-4
NUM_CLASSES = 3
NUM_WORKERS = 4

# Load data
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

# Preprocess train data
train_df["case"] = train_df["id"].apply(
    lambda x: int(x.split("_")[0].replace("case", ""))
)
train_df["day"] = train_df["id"].apply(
    lambda x: int(x.split("_")[1].replace("day", ""))
)
train_df["slice"] = train_df["id"].apply(
    lambda x: int(x.split("_")[3].replace("slice_", ""))
)


def get_image_path(row, is_train=True):
    folder = "train" if is_train else "test"
    case_str = f"case{row['case']}"
    day_str = f"case{row['case']}_day{row['day']}"
    slice_num = str(row["slice"]).zfill(4)

    case_folder = f"./input/{folder}/{case_str}/{day_str}/scans/"
    if not os.path.exists(case_folder):
        return None

    files = os.listdir(case_folder)
    for f in files:
        if f.startswith(f"slice_{slice_num}_"):
            return os.path.join(case_folder, f)
    return None


train_df["image_path"] = train_df.apply(
    lambda x: get_image_path(x, is_train=True), axis=1
)
train_df = train_df.dropna(subset=["image_path"])


def rle_decode(mask_rle, shape):
    if pd.isna(mask_rle) or mask_rle == "":
        return np.zeros(shape, dtype=np.uint8)

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        mask[lo:hi] = 1
    return mask.reshape(shape).T


def rle_encode(img):
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


# Prepare training slices
all_slices = []
for (case, day, slice_idx), group in train_df.groupby(["case", "day", "slice"]):
    image_path = get_image_path(
        {"case": case, "day": day, "slice": slice_idx}, is_train=True
    )
    if image_path and os.path.exists(image_path):
        all_slices.append(
            {"case": case, "day": day, "slice": slice_idx, "image_path": image_path}
        )

slices_df = pd.DataFrame(all_slices)


class GI_Dataset(Dataset):
    def __init__(self, slices_df, transform=None, is_train=True):
        self.slices_df = slices_df.reset_index(drop=True)
        self.transform = transform
        self.is_train = is_train

    def __len__(self):
        return len(self.slices_df)

    def __getitem__(self, idx):
        row = self.slices_df.iloc[idx]
        case = row["case"]
        day = row["day"]
        slice_idx = row["slice"]
        image_path = row["image_path"]

        # Load and preprocess image
        try:
            image = cv2.imread(image_path, cv2.IMREAD_ANYDEPTH)
            if image is None:
                image = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint16)
        except:
            image = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint16)

        # Get original dimensions from filename
        filename = os.path.basename(image_path)
        parts = filename.split("_")
        if len(parts) >= 3:
            try:
                original_h, original_w = int(parts[2]), int(parts[1])
            except:
                original_h, original_w = IMG_SIZE, IMG_SIZE
        else:
            original_h, original_w = IMG_SIZE, IMG_SIZE

        # Resize and normalize
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        if image.max() > image.min():
            image = (image - image.min()) / (image.max() - image.min() + 1e-6)
        image = (image * 255).astype(np.uint8)

        if self.is_train:
            # Load and combine masks for all classes
            mask = np.zeros((IMG_SIZE, IMG_SIZE, NUM_CLASSES), dtype=np.uint8)
            for class_idx, class_name in enumerate(
                ["large_bowel", "small_bowel", "stomach"]
            ):
                mask_row = train_df[
                    (train_df["case"] == case)
                    & (train_df["day"] == day)
                    & (train_df["slice"] == slice_idx)
                    & (train_df["class"] == class_name)
                ]

                if not mask_row.empty and not pd.isna(mask_row["segmentation"].iloc[0]):
                    mask_rle = mask_row["segmentation"].iloc[0]
                    class_mask = rle_decode(mask_rle, (original_h, original_w))
                    class_mask = cv2.resize(class_mask, (IMG_SIZE, IMG_SIZE))
                    mask[:, :, class_idx] = class_mask

            # Apply transforms
            if self.transform:
                augmented = self.transform(image=image, mask=mask)
                image = augmented["image"]
                mask = augmented["mask"]

            # Convert to tensors
            image_tensor = torch.from_numpy(image).float().unsqueeze(0)
            mask_tensor = torch.from_numpy(mask).permute(2, 0, 1).float()

            return image_tensor, mask_tensor
        else:
            # Test mode
            if self.transform:
                augmented = self.transform(image=image)
                image = augmented["image"]

            image_tensor = torch.from_numpy(image).float().unsqueeze(0)
            return image_tensor, f"case{case}_day{day}_slice_{str(slice_idx).zfill(4)}"


# 2D U-Net Model
class UNet2D(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder
        for feature in features:
            self.encoder.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, feature, kernel_size=3, padding=1),
                    nn.BatchNorm2d(feature),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(feature, feature, kernel_size=3, padding=1),
                    nn.BatchNorm2d(feature),
                    nn.ReLU(inplace=True),
                )
            )
            in_channels = feature

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features[-1], features[-1] * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(features[-1] * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[-1] * 2, features[-1] * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(features[-1] * 2),
            nn.ReLU(inplace=True),
        )

        # Decoder
        features = features[::-1]
        for i, feature in enumerate(features):
            self.decoder.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.decoder.append(
                nn.Sequential(
                    nn.Conv2d(feature * 2, feature, kernel_size=3, padding=1),
                    nn.BatchNorm2d(feature),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(feature, feature, kernel_size=3, padding=1),
                    nn.BatchNorm2d(feature),
                    nn.ReLU(inplace=True),
                )
            )

        # Final layer
        self.final = nn.Conv2d(features[-1], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Encoder
        for encode in self.encoder:
            x = encode(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = F.interpolate(
                    x,
                    size=skip_connection.shape[2:],
                    mode="bilinear",
                    align_corners=True,
                )

            x = torch.cat((skip_connection, x), dim=1)
            x = self.decoder[idx + 1](x)

        return torch.sigmoid(self.final(x))


# Loss function
class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = pred.contiguous()
        target = target.contiguous()

        # Dice loss
        intersection = (pred * target).sum(dim=(2, 3))
        dice = (2.0 * intersection + self.smooth) / (
            pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + self.smooth
        )
        dice_loss = 1 - dice.mean(dim=1)

        # BCE loss
        bce = F.binary_cross_entropy(pred, target, reduction="none").mean(dim=(1, 2, 3))

        return (dice_loss + bce).mean()


def dice_score(pred, target, smooth=1e-6):
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    return (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for images, masks in tqdm(loader, desc="Training"):
        images = images.float().to(device)
        masks = masks.float().to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, device):
    model.eval()
    dice_scores = []

    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Validation"):
            images = images.float().to(device)
            masks = masks.float().to(device)

            outputs = model(images)

            for i in range(outputs.shape[0]):
                for c in range(outputs.shape[1]):
                    dice = dice_score(outputs[i, c], masks[i, c])
                    dice_scores.append(dice.item())

    return np.mean(dice_scores)


print("Creating cross-validation folds...")
groups = slices_df["case"].values
gkf = GroupKFold(n_splits=3)

best_val_score = 0
best_model_state = None

# Train model
for fold, (train_idx, val_idx) in enumerate(gkf.split(slices_df, groups=groups)):
    if fold >= 1:
        break

    print(f"\nFold {fold + 1}")

    train_slices = slices_df.iloc[train_idx]
    val_slices = slices_df.iloc[val_idx]

    print(
        f"Training samples: {len(train_slices)}, Validation samples: {len(val_slices)}"
    )

    train_transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.5
            ),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
        ]
    )

    val_transform = A.Compose([])

    train_dataset = GI_Dataset(train_slices, transform=train_transform, is_train=True)
    val_dataset = GI_Dataset(val_slices, transform=val_transform, is_train=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    model = UNet2D(in_channels=1, out_channels=NUM_CLASSES).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )
    criterion = DiceBCELoss()

    best_fold_score = 0
    current_lr = LR

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        val_dice = validate(model, val_loader, device)

        # Check if learning rate changed
        old_lr = current_lr
        scheduler.step(val_dice)
        current_lr = optimizer.param_groups[0]["lr"]
        if current_lr != old_lr:
            print(f"Learning rate reduced to: {current_lr}")

        print(f"Train Loss: {train_loss:.4f}, Val Dice: {val_dice:.4f}")

        if val_dice > best_fold_score:
            best_fold_score = val_dice
            model_path = f"./working/best_model_fold{fold}.pth"
            torch.save(model.state_dict(), model_path)
            best_model_state = model.state_dict().copy()

    print(f"Fold {fold+1} completed. Best dice score: {best_fold_score:.4f}")

print(f"\nLoading best model...")
model = UNet2D(in_channels=1, out_channels=NUM_CLASSES).to(device)
model.load_state_dict(best_model_state)
model.eval()

# Final validation
print("\n=== Final Model Validation ===")
val_dice = validate(model, val_loader, device)
best_val_score = val_dice

print(f"\nFinal Validation Dice Score: {best_val_score:.4f}")

# Prepare test data
print("\nPreparing test data...")
test_slices_info = []
for idx, row in test_df.iterrows():
    id_str = row["id"]
    parts = id_str.split("_")
    case = int(parts[0].replace("case", ""))
    day = int(parts[1].replace("day", ""))
    slice_idx = int(parts[3])

    test_slices_info.append(
        {"case": case, "day": day, "slice": slice_idx, "class": row["class"]}
    )

test_slices_df = pd.DataFrame(test_slices_info)
unique_test_images = (
    test_slices_df[["case", "day", "slice"]].drop_duplicates().reset_index(drop=True)
)


# Get image paths with fallback
def get_test_image_path(row):
    path = get_image_path(row, is_train=False)
    if path and os.path.exists(path):
        return path
    else:
        # Fallback: check if there's any image in the directory
        folder = "test"
        case_str = f"case{row['case']}"
        day_str = f"case{row['case']}_day{row['day']}"
        slice_num = str(row["slice"]).zfill(4)

        case_folder = f"./input/{folder}/{case_str}/{day_str}/scans/"
        if os.path.exists(case_folder):
            files = [
                f
                for f in os.listdir(case_folder)
                if f.startswith(f"slice_{slice_num}_")
            ]
            if files:
                return os.path.join(case_folder, files[0])
        return None


unique_test_images["image_path"] = unique_test_images.apply(
    lambda x: get_test_image_path(x), axis=1
)

# Create test dataset even if some images are missing
test_transform = A.Compose([])
test_dataset = GI_Dataset(unique_test_images, transform=test_transform, is_train=False)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

print("Making predictions on test set...")
all_predictions = {}

with torch.no_grad():
    for images, ids in tqdm(test_loader, desc="Predicting"):
        images = images.float().to(device)
        outputs = model(images)
        outputs = outputs.cpu().numpy()

        for i in range(len(ids)):
            all_predictions[ids[i]] = outputs[i]

print("Creating submission file...")
submission_rows = []

for idx, row in tqdm(
    test_df.iterrows(), total=len(test_df), desc="Encoding predictions"
):
    id_str = row["id"]
    class_name = row["class"]

    parts = id_str.split("_")
    case = int(parts[0].replace("case", ""))
    day = int(parts[1].replace("day", ""))
    slice_idx = int(parts[3])
    slice_id = f"case{case}_day{day}_slice_{str(slice_idx).zfill(4)}"

    if slice_id in all_predictions:
        pred = all_predictions[slice_id]

        class_to_idx = {"large_bowel": 0, "small_bowel": 1, "stomach": 2}
        class_idx = class_to_idx[class_name]

        mask = pred[class_idx]

        # Get original dimensions
        original_h, original_w = IMG_SIZE, IMG_SIZE
        try:
            test_folder = f"./input/test/case{case}/case{case}_day{day}/scans/"
            if os.path.exists(test_folder):
                files = [
                    f
                    for f in os.listdir(test_folder)
                    if f.startswith(f"slice_{str(slice_idx).zfill(4)}_")
                ]
                if files:
                    filename = files[0]
                    parts = filename.split("_")
                    original_h = int(parts[2])
                    original_w = int(parts[1])
        except:
            pass

        mask_resized = cv2.resize(mask, (original_w, original_h))
        mask_binary = (mask_resized > 0.5).astype(np.uint8)

        rle = rle_encode(mask_binary)
    else:
        rle = ""

    submission_rows.append({"id": id_str, "class": class_name, "predicted": rle})

submission_df = pd.DataFrame(submission_rows)

# Ensure all test IDs are present
all_test_ids = set(test_df["id"])
submission_ids = set(submission_df["id"])
missing_ids = all_test_ids - submission_ids

if len(missing_ids) > 0:
    print(
        f"Warning: {len(missing_ids)} rows missing predictions, filling with empty strings"
    )
    for id_str in missing_ids:
        row = test_df[test_df["id"] == id_str].iloc[0]
        submission_df = pd.concat(
            [
                submission_df,
                pd.DataFrame([{"id": id_str, "class": row["class"], "predicted": ""}]),
            ],
            ignore_index=True,
        )

os.makedirs("./submission", exist_ok=True)
submission_path = "./submission/submission.csv"
submission_df.to_csv(submission_path, index=False)

print(f"\n=== SUBMISSION CREATED ===")
print(f"File saved to: {submission_path}")
print(f"File size: {os.path.getsize(submission_path) / 1024:.2f} KB")
print(f"Number of predictions: {len(submission_df)}")

print(f"\n=== FINAL VALIDATION METRIC ===")
print(f"2D Model Dice Score: {best_val_score:.6f}")

print("\n=== SUBMISSION VALIDATION ===")
sample_submission = pd.read_csv("./input/sample_submission.csv")
if len(submission_df) == len(sample_submission):
    print(f"✓ Submission has correct number of rows: {len(submission_df)}")
else:
    print(
        f"✗ Submission has {len(submission_df)} rows, expected {len(sample_submission)}"
    )

if list(submission_df.columns) == ["id", "class", "predicted"]:
    print("✓ Submission has correct columns")
else:
    print(f"✗ Submission columns: {list(submission_df.columns)}")

missing_ids = set(sample_submission["id"]) - set(submission_df["id"])
if len(missing_ids) == 0:
    print("✓ All required IDs are present")
else:
    print(f"✗ Missing {len(missing_ids)} IDs")

# Cleanup
import glob

for f in glob.glob("./working/*.pth"):
    try:
        os.remove(f)
    except:
        pass

print("\n=== SEGMENTATION PROCESS COMPLETED SUCCESSFULLY ===")
