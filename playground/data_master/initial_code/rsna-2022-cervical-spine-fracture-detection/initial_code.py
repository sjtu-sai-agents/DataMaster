import os
import numpy as np
import pandas as pd
import pydicom
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Paths
INPUT_DIR = "./input"
TRAIN_CSV = os.path.join(INPUT_DIR, "train.csv")
TEST_CSV = os.path.join(INPUT_DIR, "test.csv")
TRAIN_IMAGES_DIR = os.path.join(INPUT_DIR, "train_images")
TEST_IMAGES_DIR = os.path.join(INPUT_DIR, "test_images")
SUBMISSION_DIR = "./submission"
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Load data
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

# Prepare targets
target_cols = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "patient_overall"]
train_df["StudyInstanceUID"] = train_df["StudyInstanceUID"].astype(str)
test_df["StudyInstanceUID"] = test_df["StudyInstanceUID"].astype(str)

# Create validation split (stratified by patient_overall)
train_studies, val_studies = train_test_split(
    train_df["StudyInstanceUID"].unique(),
    test_size=0.2,
    random_state=42,
    stratify=train_df.set_index("StudyInstanceUID")["patient_overall"].reindex(
        train_df["StudyInstanceUID"].unique()
    ),
)


# Updated Dataset class with train/test mode
class CervicalSpineDataset(Dataset):
    def __init__(
        self, study_uids, df, images_dir, transform=None, num_slices=20, is_train=True
    ):
        self.study_uids = study_uids
        self.df = df.set_index("StudyInstanceUID")
        self.images_dir = images_dir
        self.transform = transform
        self.num_slices = num_slices
        self.windows = [
            (400, 1800),
            (50, 350),
            (600, 2800),
        ]  # bone, soft tissue, custom
        self.is_train = is_train
        if self.is_train:
            self.target_cols = target_cols

    def __len__(self):
        return len(self.study_uids)

    def _apply_window(self, image, center, width):
        lower = center - width // 2
        upper = center + width // 2
        windowed = np.clip(image, lower, upper)
        windowed = (windowed - lower) / (upper - lower + 1e-5)
        return windowed

    def _load_slices(self, study_uid):
        study_path = os.path.join(self.images_dir, study_uid)
        slice_files = sorted(
            [f for f in os.listdir(study_path) if f.endswith(".dcm")],
            key=lambda x: int(x.split(".")[0]),
        )
        if len(slice_files) == 0:
            raise ValueError(f"No DICOM files found for {study_uid}")
        # Sample slices uniformly
        indices = np.linspace(0, len(slice_files) - 1, self.num_slices, dtype=int)
        selected_files = [slice_files[i] for i in indices]
        slices = []
        for fname in selected_files:
            dicom_path = os.path.join(study_path, fname)
            dicom = pydicom.dcmread(dicom_path)
            image = dicom.pixel_array.astype(np.float32)
            # Apply rescale intercept and slope if present
            intercept = dicom.get("RescaleIntercept", 0.0)
            slope = dicom.get("RescaleSlope", 1.0)
            image = image * slope + intercept
            # Apply three different windows
            windowed_images = []
            for center, width in self.windows:
                windowed = self._apply_window(image, center, width)
                windowed_images.append(windowed)
            # Stack along channel dimension (3 channels per slice)
            combined = np.stack(windowed_images, axis=0)  # (3, H, W)
            slices.append(combined)
        # Stack slices: (num_slices, 3, H, W)
        volume = np.stack(slices, axis=0)
        return volume

    def __getitem__(self, idx):
        study_uid = self.study_uids[idx]
        volume = self._load_slices(study_uid)  # (S, C, H, W)
        # Apply transform if provided
        if self.transform:
            # Transform each slice individually
            transformed_slices = []
            for i in range(volume.shape[0]):
                slice_img = volume[i]  # (C, H, W)
                slice_img = self.transform(torch.tensor(slice_img))
                transformed_slices.append(slice_img)
            volume = torch.stack(transformed_slices, dim=0)
        if self.is_train:
            # Get labels for training
            labels = self.df.loc[study_uid, self.target_cols].values.astype(np.float32)
            return volume, torch.tensor(labels), study_uid
        else:
            # For test, return only volume and study_uid
            return volume, study_uid


# Model
class SiameseSpineModel(nn.Module):
    def __init__(self, num_classes=8, num_slices=20, feature_dim=512):
        super().__init__()
        self.num_slices = num_slices
        # Three separate ResNet18 backbones (one for each window)
        self.backbones = nn.ModuleList(
            [
                models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
                for _ in range(3)
            ]
        )
        # Remove the final fully connected layer from each backbone
        for backbone in self.backbones:
            backbone.fc = nn.Identity()
        # Feature reduction for each backbone
        self.feature_reductions = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(512, feature_dim), nn.ReLU(), nn.Dropout(0.3))
                for _ in range(3)
            ]
        )
        # Slice aggregation (pool across slices)
        self.slice_pool = nn.AdaptiveAvgPool1d(1)
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(3 * feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        # x shape: (batch, slices, channels=3, H, W)
        batch_size, num_slices, C, H, W = x.shape
        # Process each window channel separately
        window_features = []
        for window_idx in range(3):
            # Extract the channel corresponding to this window
            window_data = x[:, :, window_idx, :, :]  # (batch, slices, H, W)
            window_data = window_data.reshape(batch_size * num_slices, 1, H, W)
            # Repeat channel to 3 for ResNet input
            window_data = window_data.repeat(1, 3, 1, 1)
            # Forward through backbone
            features = self.backbones[window_idx](window_data)  # (batch*slices, 512)
            features = features.reshape(
                batch_size, num_slices, -1
            )  # (batch, slices, 512)
            # Reduce features
            features = self.feature_reductions[window_idx](
                features
            )  # (batch, slices, feature_dim)
            # Pool across slices
            features = features.transpose(1, 2)  # (batch, feature_dim, slices)
            features = self.slice_pool(features).squeeze(-1)  # (batch, feature_dim)
            window_features.append(features)
        # Concatenate window features
        combined = torch.cat(window_features, dim=1)  # (batch, 3*feature_dim)
        # Classifier
        out = self.classifier(combined)
        return torch.sigmoid(out)


# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_slices = 20
batch_size = 4
num_workers = 8
epochs = 10

# Transforms
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize each channel
    ]
)

# Datasets and dataloaders
train_dataset = CervicalSpineDataset(
    train_studies, train_df, TRAIN_IMAGES_DIR, transform, num_slices, is_train=True
)
val_dataset = CervicalSpineDataset(
    val_studies, train_df, TRAIN_IMAGES_DIR, transform, num_slices, is_train=True
)

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
)

# Model, loss, optimizer
model = SiameseSpineModel(num_classes=8, num_slices=num_slices, feature_dim=256).to(
    device
)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", patience=2, factor=0.5
)

# Training loop
best_val_loss = float("inf")
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
        volumes, labels, _ = batch
        volumes, labels = volumes.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(volumes)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * volumes.size(0)
    train_loss /= len(train_dataset)

    # Validation
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation"):
            volumes, labels, _ = batch
            volumes, labels = volumes.to(device), labels.to(device)
            outputs = model(volumes)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * volumes.size(0)
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    val_loss /= len(val_dataset)
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Compute weighted log loss (competition metric)
    # Weights: patient_overall has weight 2, others weight 1 (as per competition)
    weights = np.array([1, 1, 1, 1, 1, 1, 1, 2])
    sample_weights = np.tile(weights, (all_preds.shape[0], 1))
    val_log_loss = log_loss(
        all_labels.flatten(),
        all_preds.flatten(),
        sample_weight=sample_weights.flatten(),
    )

    print(
        f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Weighted Log Loss: {val_log_loss:.4f}"
    )

    scheduler.step(val_log_loss)
    if val_log_loss < best_val_loss:
        best_val_loss = val_log_loss
        torch.save(model.state_dict(), "./working/best_model.pth")

print(f"Best validation weighted log loss: {best_val_loss:.4f}")

# Prepare test dataset (is_train=False)
test_studies = test_df["StudyInstanceUID"].unique()
test_dataset = CervicalSpineDataset(
    test_studies, test_df, TEST_IMAGES_DIR, transform, num_slices, is_train=False
)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
)

# Load best model and predict on test set
model.load_state_dict(torch.load("./working/best_model.pth", map_location=device))
model.eval()
test_preds = {}
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Predicting on test set"):
        if len(batch) == 3:  # Should not happen with is_train=False, but just in case
            volumes, labels, study_uids = batch
        else:
            volumes, study_uids = batch
        volumes = volumes.to(device)
        outputs = model(volumes)
        for i, uid in enumerate(study_uids):
            test_preds[uid] = outputs[i].cpu().numpy()

# Create submission DataFrame
type_to_idx = {col: i for i, col in enumerate(target_cols)}
submission_rows = []
for _, row in test_df.iterrows():
    study_uid = row["StudyInstanceUID"]
    pred_type = row["prediction_type"]
    if study_uid in test_preds:
        preds = test_preds[study_uid]
        fractured = preds[type_to_idx[pred_type]]
    else:
        fractured = 0.0  # fallback
    row_id = f"{study_uid}_{pred_type}"
    submission_rows.append({"row_id": row_id, "fractured": fractured})

submission_df = pd.DataFrame(submission_rows)
submission_df.to_csv(SUBMISSION_PATH, index=False)
print(f"Submission saved to {SUBMISSION_PATH}")
print(f"Submission shape: {submission_df.shape}")
print(f"Validation weighted log loss: {best_val_loss:.4f}")
