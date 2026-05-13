import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import timm
from sklearn.metrics import roc_curve, auc
import warnings
import cv2
from tqdm import tqdm
from collections import OrderedDict

warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Configuration
DATA_DIR = "./input"
COVER_DIR = os.path.join(DATA_DIR, "Cover")
JMIPOD_DIR = os.path.join(DATA_DIR, "JMiPOD")
JUNIWARD_DIR = os.path.join(DATA_DIR, "JUNIWARD")
UERD_DIR = os.path.join(DATA_DIR, "UERD")
TEST_DIR = os.path.join(DATA_DIR, "Test")
SUBMISSION_DIR = "./submission"
WORKING_DIR = "./working"

os.makedirs(SUBMISSION_DIR, exist_ok=True)
os.makedirs(WORKING_DIR, exist_ok=True)

# Hyperparameters
BATCH_SIZE = 64
EPOCHS = 5
LR = 1e-4
IMG_SIZE = 224
NUM_WORKERS = 8
VALIDATION_SPLIT = 0.2

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Prepare dataset
print("Preparing dataset...")

# Collect all image paths and labels
image_paths = []
labels = []
cover_ids = []

# Cover images (label 0)
for fname in os.listdir(COVER_DIR):
    if fname.endswith(".jpg"):
        image_paths.append(os.path.join(COVER_DIR, fname))
        labels.append(0)
        cover_ids.append(fname.split(".")[0])

# Stego images (label 1) - combine all three types
stego_dirs = [JMIPOD_DIR, JUNIWARD_DIR, UERD_DIR]
for stego_dir in stego_dirs:
    for fname in os.listdir(stego_dir):
        if fname.endswith(".jpg"):
            image_paths.append(os.path.join(stego_dir, fname))
            labels.append(1)
            cover_ids.append(fname.split(".")[0])

# Create DataFrame
df = pd.DataFrame({"image_path": image_paths, "label": labels, "cover_id": cover_ids})

# Split by cover_id to avoid data leakage
unique_ids = df["cover_id"].unique()
np.random.shuffle(unique_ids)
split_idx = int(len(unique_ids) * (1 - VALIDATION_SPLIT))
train_ids = unique_ids[:split_idx]
val_ids = unique_ids[split_idx:]

train_df = df[df["cover_id"].isin(train_ids)].reset_index(drop=True)
val_df = df[df["cover_id"].isin(val_ids)].reset_index(drop=True)

print(f"Training samples: {len(train_df)}, Validation samples: {len(val_df)}")

# Define SRM (Spatial Rich Model) kernels - commonly used in steganalysis
SRM_KERNELS = [
    np.array(
        [
            [0, 0, 0, 0, 0],
            [0, -1, 2, -1, 0],
            [0, 2, -4, 2, 0],
            [0, -1, 2, -1, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.float32,
    ),
    np.array(
        [
            [-1, 2, -2, 2, -1],
            [2, -6, 8, -6, 2],
            [-2, 8, -12, 8, -2],
            [2, -6, 8, -6, 2],
            [-1, 2, -2, 2, -1],
        ],
        dtype=np.float32,
    ),
    np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 1, -4, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.float32,
    ),
]


def extract_gaussian_residual(image, kernel_size=5):
    """Extract noise residual using Gaussian blur subtraction"""
    if isinstance(image, np.ndarray):
        img_np = image
    else:
        img_np = np.array(image)

    blurred = cv2.GaussianBlur(img_np, (kernel_size, kernel_size), 0)
    residual = img_np.astype(np.float32) - blurred.astype(np.float32)
    residual = residual / 127.5
    residual = np.clip(residual, -1.0, 1.0)
    residual = (residual * 127.5 + 127.5).astype(np.uint8)
    return Image.fromarray(residual)


def extract_srm_residual(image):
    """Extract noise residual using SRM filters"""
    if isinstance(image, np.ndarray):
        img_np = image
    else:
        img_np = np.array(image)

    # Convert to grayscale for SRM filters
    if len(img_np.shape) == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_np

    residuals = []
    for kernel in SRM_KERNELS:
        filtered = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        residuals.append(filtered)

    # Combine residuals and normalize
    residual = np.mean(residuals, axis=0)
    residual = residual / np.std(residual) if np.std(residual) > 0 else residual
    residual = np.clip(residual, -3, 3)
    residual = ((residual + 3) / 6 * 255).astype(np.uint8)

    # Convert back to 3-channel
    residual = np.stack([residual] * 3, axis=2)
    return Image.fromarray(residual)


def extract_dct_residual(image):
    """Extract residual using DCT-based approach"""
    if isinstance(image, np.ndarray):
        img_np = image
    else:
        img_np = np.array(image)

    # Convert to YCbCr and work on Y channel
    if len(img_np.shape) == 3:
        ycbcr = cv2.cvtColor(img_np, cv2.COLOR_RGB2YCrCb)
        y_channel = ycbcr[:, :, 0].astype(np.float32)
    else:
        y_channel = img_np.astype(np.float32)

    # Block DCT (8x8 blocks)
    h, w = y_channel.shape
    residual = np.zeros_like(y_channel)

    for i in range(0, h - 7, 8):
        for j in range(0, w - 7, 8):
            block = y_channel[i : i + 8, j : j + 8]
            dct_block = cv2.dct(block)
            # Keep only high-frequency components
            dct_block[:4, :4] = 0
            residual_block = cv2.idct(dct_block)
            residual[i : i + 8, j : j + 8] = residual_block

    residual = residual / np.std(residual) if np.std(residual) > 0 else residual
    residual = np.clip(residual, -3, 3)
    residual = ((residual + 3) / 6 * 255).astype(np.uint8)

    # Convert back to 3-channel
    residual = np.stack([residual] * 3, axis=2)
    return Image.fromarray(residual)


# Create transforms for different residual methods
train_transforms = {
    "gaussian": transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.Lambda(lambda x: extract_gaussian_residual(x)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    ),
    "srm": transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.Lambda(lambda x: extract_srm_residual(x)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    ),
    "dct": transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.Lambda(lambda x: extract_dct_residual(x)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    ),
}

val_transforms = {
    "gaussian": transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.Lambda(lambda x: extract_gaussian_residual(x)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    ),
    "srm": transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.Lambda(lambda x: extract_srm_residual(x)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    ),
    "dct": transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.Lambda(lambda x: extract_dct_residual(x)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    ),
}


# Dataset class for different transforms
class StegoDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img_path = row["image_path"]
        label = row["label"]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)


# Create datasets for different residual methods
print("Creating datasets for different residual methods...")
train_datasets = {}
val_datasets = {}
train_loaders = {}
val_loaders = {}

for method in ["gaussian", "srm", "dct"]:
    train_datasets[method] = StegoDataset(train_df, transform=train_transforms[method])
    val_datasets[method] = StegoDataset(val_df, transform=val_transforms[method])

    train_loaders[method] = DataLoader(
        train_datasets[method],
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loaders[method] = DataLoader(
        val_datasets[method],
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )


# Model creation function
def create_model(model_name="efficientnet_b0"):
    model = timm.create_model(model_name, pretrained=True, num_classes=1)
    return model.to(device)


# Weighted AUC calculation
def weighted_auc(y_true, y_pred):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    tpr_thresholds = [0.0, 0.4, 1.0]
    weights = [2, 1]

    total_auc = 0
    total_weight = 0

    for i in range(len(weights)):
        mask = (tpr >= tpr_thresholds[i]) & (tpr <= tpr_thresholds[i + 1])
        if np.sum(mask) < 2:
            continue

        segment_fpr = fpr[mask]
        segment_tpr = tpr[mask]

        if tpr_thresholds[i] not in segment_tpr:
            segment_fpr = np.insert(
                segment_fpr, 0, np.interp(tpr_thresholds[i], tpr, fpr)
            )
            segment_tpr = np.insert(segment_tpr, 0, tpr_thresholds[i])
        if tpr_thresholds[i + 1] not in segment_tpr:
            segment_fpr = np.append(
                segment_fpr, np.interp(tpr_thresholds[i + 1], tpr, fpr)
            )
            segment_tpr = np.append(segment_tpr, tpr_thresholds[i + 1])

        segment_auc = auc(segment_fpr, segment_tpr)
        total_auc += segment_auc * weights[i]
        total_weight += weights[i]

    return total_auc / total_weight if total_weight > 0 else 0.0


# Train models for different residual methods
models = {}
best_val_aucs = {}

for method in ["gaussian", "srm", "dct"]:
    print(f"\nTraining model with {method} residuals...")

    model = create_model()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_auc = 0.0

    for epoch in range(EPOCHS):
        # Training
        model.train()
        train_loss = 0.0
        train_bar = tqdm(
            train_loaders[method], desc=f"{method} Epoch {epoch+1}/{EPOCHS} [Train]"
        )

        for batch_idx, (images, labels) in enumerate(train_bar):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())

        # Validation
        model.eval()
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for images, labels in tqdm(
                val_loaders[method], desc=f"{method} Epoch {epoch+1}/{EPOCHS} [Val]"
            ):
                images = images.to(device)
                outputs = model(images).squeeze()
                val_preds.extend(torch.sigmoid(outputs).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_auc = weighted_auc(val_labels, val_preds)
        print(
            f"{method} Epoch {epoch+1}, Training Loss: {train_loss/len(train_loaders[method]):.4f}, Validation Weighted AUC: {val_auc:.4f}"
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(
                model.state_dict(),
                os.path.join(WORKING_DIR, f"best_model_{method}.pth"),
            )

        scheduler.step()

    models[method] = model
    best_val_aucs[method] = best_val_auc
    print(f"Best {method} Validation Weighted AUC: {best_val_auc:.4f}")

# Load best models
for method in models.keys():
    models[method].load_state_dict(
        torch.load(os.path.join(WORKING_DIR, f"best_model_{method}.pth"))
    )
    models[method].eval()

# Optimize ensemble weights using validation data
print("\nOptimizing ensemble weights...")
val_predictions = {method: [] for method in models.keys()}
val_labels_list = []

# Collect predictions on validation set
for method in models.keys():
    model = models[method]
    model.eval()

    with torch.no_grad():
        for images, labels in tqdm(
            val_loaders[method], desc=f"Collecting {method} predictions"
        ):
            images = images.to(device)
            outputs = model(images).squeeze()
            val_predictions[method].extend(torch.sigmoid(outputs).cpu().numpy())

    val_predictions[method] = np.array(val_predictions[method])

# Get labels (same for all methods)
val_labels_list = val_df["label"].values[: len(val_predictions["gaussian"])]

# Try different weight combinations and find optimal weights
best_weights = None
best_auc = 0

# Grid search for optimal weights
weight_candidates = np.linspace(0, 1, 11)
for w1 in weight_candidates:
    for w2 in weight_candidates:
        w3 = 1.0 - w1 - w2
        if w3 >= 0:
            weights = np.array([w1, w2, w3])
            weights = weights / weights.sum()  # Normalize

            # Weighted ensemble prediction
            methods = list(models.keys())
            ensemble_pred = np.zeros_like(val_predictions[methods[0]])
            for i, method in enumerate(methods):
                ensemble_pred += weights[i] * val_predictions[method]

            ensemble_auc = weighted_auc(val_labels_list, ensemble_pred)

            if ensemble_auc > best_auc:
                best_auc = ensemble_auc
                best_weights = weights

print(f"Optimal weights: {dict(zip(models.keys(), best_weights))}")
print(f"Ensemble Validation Weighted AUC: {best_auc:.4f}")

# Prepare test data with different transforms
test_image_paths = []
test_ids = []
for fname in sorted(os.listdir(TEST_DIR)):
    if fname.endswith(".jpg"):
        test_image_paths.append(os.path.join(TEST_DIR, fname))
        test_ids.append(fname)

# Test transforms
test_transforms = {
    "gaussian": transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.Lambda(lambda x: extract_gaussian_residual(x)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    ),
    "srm": transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.Lambda(lambda x: extract_srm_residual(x)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    ),
    "dct": transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.Lambda(lambda x: extract_dct_residual(x)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    ),
}


class TestDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image


# Generate predictions for each method
test_predictions = {method: [] for method in models.keys()}

for method in models.keys():
    print(f"\nGenerating test predictions with {method} model...")

    test_dataset = TestDataset(test_image_paths, transform=test_transforms[method])
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    model = models[method]
    model.eval()

    with torch.no_grad():
        for images in tqdm(test_loader, desc=f"Predicting with {method}"):
            images = images.to(device)
            outputs = model(images).squeeze()
            test_predictions[method].extend(torch.sigmoid(outputs).cpu().numpy())

    test_predictions[method] = np.array(test_predictions[method])

# Create weighted ensemble predictions
print("\nCreating ensemble predictions...")
ensemble_preds = np.zeros_like(test_predictions["gaussian"])

for i, method in enumerate(models.keys()):
    ensemble_preds += best_weights[i] * test_predictions[method]

# Create submission file
submission = pd.DataFrame({"Id": test_ids, "Label": ensemble_preds})
submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")

# Print final validation metric
print(f"\nFinal Ensemble Validation Weighted AUC: {best_auc:.4f}")
print(f"Individual model AUCs:")
for method, auc_val in best_val_aucs.items():
    print(f"  {method}: {auc_val:.4f}")
