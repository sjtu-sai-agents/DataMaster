import json
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss
import warnings
import time

warnings.filterwarnings("ignore")

# Set seeds
torch.manual_seed(42)
np.random.seed(42)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load data
input_dir = Path("./input")
train_path = input_dir / "train.json"
test_path = input_dir / "test.json"

with open(train_path, "r") as f:
    train_data = json.load(f)
with open(test_path, "r") as f:
    test_data = json.load(f)

train_df = pd.DataFrame(train_data)
test_df = pd.DataFrame(test_data)


# Process incidence angle
def process_inc_angle(df):
    df = df.copy()
    df["inc_angle"] = df["inc_angle"].replace("na", np.nan)
    df["inc_angle"] = pd.to_numeric(df["inc_angle"], errors="coerce")
    return df


train_df = process_inc_angle(train_df)
test_df = process_inc_angle(test_df)

# Impute missing angles
angle_imputer = SimpleImputer(strategy="median")
train_df["inc_angle"] = angle_imputer.fit_transform(train_df[["inc_angle"]])
test_df["inc_angle"] = angle_imputer.transform(test_df[["inc_angle"]])

# Normalize incidence angle
angle_scaler = StandardScaler()
train_df["inc_angle"] = angle_scaler.fit_transform(train_df[["inc_angle"]])
test_df["inc_angle"] = angle_scaler.transform(test_df[["inc_angle"]])


# Normalize band values per image
def normalize_bands(df):
    for band in ["band_1", "band_2"]:
        band_data = np.array(df[band].tolist())
        # Reshape to (n_samples, 75, 75)
        band_data = band_data.reshape(-1, 75, 75)
        # Normalize each image individually
        means = band_data.mean(axis=(1, 2), keepdims=True)
        stds = band_data.std(axis=(1, 2), keepdims=True) + 1e-8
        band_data = (band_data - means) / stds
        df[band] = list(band_data.reshape(-1, 5625))
    return df


train_df = normalize_bands(train_df)
test_df = normalize_bands(test_df)


# Dataset class
class RadarDataset(Dataset):
    def __init__(self, df, is_train=True, transform=None):
        self.df = df
        self.is_train = is_train
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        band1 = np.array(self.df.iloc[idx]["band_1"]).reshape(75, 75).astype(np.float32)
        band2 = np.array(self.df.iloc[idx]["band_2"]).reshape(75, 75).astype(np.float32)
        # Stack bands to create 2-channel image
        image = np.stack([band1, band2], axis=0)  # Shape: (2, 75, 75)

        inc_angle = torch.tensor([self.df.iloc[idx]["inc_angle"]], dtype=torch.float32)

        if self.transform:
            image = self.transform(torch.tensor(image))
        else:
            image = torch.tensor(image)

        if self.is_train:
            label = torch.tensor(self.df.iloc[idx]["is_iceberg"], dtype=torch.float32)
            return image, inc_angle, label
        else:
            return image, inc_angle, self.df.iloc[idx]["id"]


# CNN with angle input
class RadarCNN(nn.Module):
    def __init__(self, base_model_name="resnet18"):
        super(RadarCNN, self).__init__()

        if base_model_name == "resnet18":
            base_model = models.resnet18(weights=None)
            # Modify first conv layer for 2 input channels
            base_model.conv1 = nn.Conv2d(
                2, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            num_features = base_model.fc.in_features
        elif base_model_name == "efficientnet_b0":
            base_model = models.efficientnet_b0(weights=None)
            base_model.features[0][0] = nn.Conv2d(
                2, 32, kernel_size=3, stride=2, padding=1, bias=False
            )
            num_features = base_model.classifier[1].in_features
        else:
            raise ValueError(f"Unknown model: {base_model_name}")

        self.cnn = base_model

        # Remove the final classification layer
        if base_model_name == "resnet18":
            self.cnn = nn.Sequential(*list(base_model.children())[:-1])
        elif base_model_name == "efficientnet_b0":
            self.cnn = nn.Sequential(*list(base_model.children())[:-1])

        # Combined classifier
        self.classifier = nn.Sequential(
            nn.Linear(num_features + 1, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, angle):
        features = self.cnn(x)
        if isinstance(features, torch.Tensor):
            features = features.view(features.size(0), -1)
        else:
            # Handle output from EfficientNet
            features = features.flatten(start_dim=1)

        combined = torch.cat([features, angle], dim=1)
        output = self.classifier(combined)
        return output.squeeze(-1)  # Only remove the last dimension if it's 1


# Data augmentation
train_transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(10),
    ]
)

# Training parameters
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-3
N_FOLDS = 5

# Prepare data
X = train_df
y = train_df["is_iceberg"].values

# Stratified K-Fold
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

# Store predictions
test_preds = []
val_scores = []

# Ensemble of different architectures
model_architectures = ["resnet18", "efficientnet_b0"]

for arch in model_architectures:
    print(f"\n=== Training {arch} ===")
    arch_test_preds = np.zeros(len(test_df))
    fold_val_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\nFold {fold + 1}/{N_FOLDS}")

        # Split data
        train_fold = train_df.iloc[train_idx].reset_index(drop=True)
        val_fold = train_df.iloc[val_idx].reset_index(drop=True)

        # Datasets
        train_dataset = RadarDataset(
            train_fold, is_train=True, transform=train_transform
        )
        val_dataset = RadarDataset(val_fold, is_train=True, transform=None)
        test_dataset = RadarDataset(test_df, is_train=False, transform=None)

        # Dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        # Model
        model = RadarCNN(base_model_name=arch).to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=3, factor=0.5
        )

        # Training
        best_val_loss = float("inf")
        for epoch in range(EPOCHS):
            model.train()
            train_loss = 0.0

            for images, angles, labels in train_loader:
                images, angles, labels = (
                    images.to(device),
                    angles.to(device),
                    labels.to(device),
                )

                optimizer.zero_grad()
                outputs = model(images, angles)
                # Ensure outputs and labels have same shape
                if outputs.dim() == 0:
                    outputs = outputs.unsqueeze(0)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * images.size(0)

            # Validation
            model.eval()
            val_loss = 0.0
            val_preds = []
            val_labels = []

            with torch.no_grad():
                for images, angles, labels in val_loader:
                    images, angles, labels = (
                        images.to(device),
                        angles.to(device),
                        labels.to(device),
                    )

                    outputs = model(images, angles)
                    # Ensure outputs and labels have same shape
                    if outputs.dim() == 0:
                        outputs = outputs.unsqueeze(0)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * images.size(0)
                    val_preds.extend(outputs.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())

            train_loss = train_loss / len(train_loader.dataset)
            val_loss = val_loss / len(val_loader.dataset)

            scheduler.step(val_loss)

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    model.state_dict(), f"./working/best_model_fold{fold}_{arch}.pth"
                )

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch+1}/{EPOCHS}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                )

        # Load best model
        model.load_state_dict(torch.load(f"./working/best_model_fold{fold}_{arch}.pth"))

        # Calculate validation log loss using sklearn
        val_log_loss = log_loss(val_labels, val_preds)
        fold_val_scores.append(val_log_loss)
        print(f"Fold {fold+1} Val Log Loss: {val_log_loss:.6f}")

        # Predict on test set
        model.eval()
        test_fold_preds = []
        test_ids = []

        with torch.no_grad():
            for images, angles, ids in test_loader:
                images, angles = images.to(device), angles.to(device)
                outputs = model(images, angles)
                if outputs.dim() == 0:
                    outputs = outputs.unsqueeze(0)
                test_fold_preds.extend(outputs.cpu().numpy())
                test_ids.extend(ids)

        # Align predictions with test_df order
        test_fold_df = pd.DataFrame({"id": test_ids, "pred": test_fold_preds})
        test_fold_df = test_fold_df.set_index("id").reindex(test_df["id"]).reset_index()
        arch_test_preds += np.array(test_fold_df["pred"]) / N_FOLDS

    # Store results for this architecture
    test_preds.append(arch_test_preds)
    val_scores.extend(fold_val_scores)
    print(
        f"{arch} average val log loss: {np.mean(fold_val_scores):.6f} (±{np.std(fold_val_scores):.6f})"
    )

# Ensemble predictions (average across architectures)
final_test_preds = np.mean(test_preds, axis=0)

# Create submission
submission_df = pd.DataFrame({"id": test_df["id"], "is_iceberg": final_test_preds})

# Clip predictions
submission_df["is_iceberg"] = submission_df["is_iceberg"].clip(0.001, 0.999)

# Save submission
submission_dir = Path("./submission")
submission_dir.mkdir(exist_ok=True)
submission_path = submission_dir / "submission.csv"
submission_df.to_csv(submission_path, index=False)

print(f"\n=== Results ===")
print(
    f"Overall validation log loss: {np.mean(val_scores):.6f} (±{np.std(val_scores):.6f})"
)
print(f"Submission saved to: {submission_path}")
print(
    f"Prediction range: [{submission_df['is_iceberg'].min():.4f}, {submission_df['is_iceberg'].max():.4f}]"
)
print(f"Sample predictions:")
print(submission_df.head(10))
