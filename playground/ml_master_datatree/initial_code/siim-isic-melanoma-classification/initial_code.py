import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import warnings

warnings.filterwarnings("ignore")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Paths
INPUT_PATH = "./input"
TRAIN_IMG_PATH = os.path.join(INPUT_PATH, "jpeg/train")
TEST_IMG_PATH = os.path.join(INPUT_PATH, "jpeg/test")
TRAIN_CSV = os.path.join(INPUT_PATH, "train.csv")
TEST_CSV = os.path.join(INPUT_PATH, "test.csv")

# Load data
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)


# Preprocess metadata
def preprocess_metadata(df, is_train=True):
    df = df.copy()
    # Fill missing values
    df["age_approx"].fillna(df["age_approx"].median(), inplace=True)
    df["sex"].fillna("unknown", inplace=True)
    df["anatom_site_general_challenge"].fillna("unknown", inplace=True)

    # Encode categorical features
    le_sex = LabelEncoder()
    le_site = LabelEncoder()

    if is_train:
        df["sex_encoded"] = le_sex.fit_transform(df["sex"])
        df["site_encoded"] = le_site.fit_transform(df["anatom_site_general_challenge"])
        # Save encoders for test set
        metadata = {
            "le_sex": le_sex,
            "le_site": le_site,
            "age_mean": df["age_approx"].mean(),
            "age_std": df["age_approx"].std(),
        }
    else:
        # Use training encoders
        le_sex = LabelEncoder()
        le_site = LabelEncoder()
        le_sex.classes_ = np.load("./working/le_sex_classes.npy", allow_pickle=True)
        le_site.classes_ = np.load("./working/le_site_classes.npy", allow_pickle=True)
        df["sex_encoded"] = le_sex.transform(df["sex"])
        df["site_encoded"] = le_site.transform(df["anatom_site_general_challenge"])
        metadata = None

    # Standardize age
    if is_train:
        age_scaler = StandardScaler()
        df["age_scaled"] = age_scaler.fit_transform(df[["age_approx"]])
        np.save("./working/age_scaler_mean.npy", age_scaler.mean_)
        np.save("./working/age_scaler_scale.npy", age_scaler.scale_)
    else:
        age_mean = np.load("./working/age_scaler_mean.npy")
        age_scale = np.load("./working/age_scaler_scale.npy")
        df["age_scaled"] = (df["age_approx"] - age_mean) / age_scale

    # Create metadata tensor
    meta_cols = ["age_scaled", "sex_encoded", "site_encoded"]
    metadata_tensor = df[meta_cols].values.astype(np.float32)

    return df, metadata_tensor, metadata


# Preprocess training metadata
train_df, train_meta, meta_encoders = preprocess_metadata(train_df, is_train=True)
# Save encoders for test set
np.save("./working/le_sex_classes.npy", meta_encoders["le_sex"].classes_)
np.save("./working/le_site_classes.npy", meta_encoders["le_site"].classes_)

# Create validation split (random split, not patient-aware for simplicity)
train_idx, val_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.2,
    random_state=42,
    stratify=train_df["target"],
)


# Dataset class
class MelanomaDataset(Dataset):
    def __init__(self, df, meta_tensor, img_dir, transform=None, is_train=True):
        self.df = df.reset_index(drop=True)
        self.meta_tensor = meta_tensor
        self.img_dir = img_dir
        self.transform = transform
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.loc[idx, "image_name"] + ".jpg"
        img_path = os.path.join(self.img_dir, img_name)

        # Load image
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Get metadata
        metadata = self.meta_tensor[idx]

        if self.is_train:
            target = self.df.loc[idx, "target"]
            return image, metadata, target
        else:
            return image, metadata


# Transformations
train_transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

val_transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# Create datasets and dataloaders
train_dataset = MelanomaDataset(
    train_df.iloc[train_idx],
    train_meta[train_idx],
    TRAIN_IMG_PATH,
    train_transform,
    is_train=True,
)
val_dataset = MelanomaDataset(
    train_df.iloc[val_idx],
    train_meta[val_idx],
    TRAIN_IMG_PATH,
    val_transform,
    is_train=True,
)

train_loader = DataLoader(
    train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
)


# Model definition
class MelanomaModel(nn.Module):
    def __init__(self, meta_dim=3):
        super(MelanomaModel, self).__init__()
        # Image branch (ResNet50)
        self.img_model = models.resnet50(pretrained=True)
        self.img_model.fc = nn.Identity()
        img_features = 2048

        # Metadata branch
        self.meta_fc = nn.Sequential(
            nn.Linear(meta_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        meta_features = 32

        # Combined classifier
        self.classifier = nn.Sequential(
            nn.Linear(img_features + meta_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img, meta):
        # Image features
        img_features = self.img_model(img)

        # Metadata features
        meta_features = self.meta_fc(meta)

        # Concatenate and classify
        combined = torch.cat([img_features, meta_features], dim=1)
        output = self.classifier(combined)
        return output.squeeze()


# Initialize model, loss, optimizer
model = MelanomaModel().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", patience=2, factor=0.5
)

# Training loop
num_epochs = 5
best_val_auc = 0

for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0
    train_preds = []
    train_targets = []

    for images, metadata, targets in train_loader:
        images = images.to(device)
        metadata = metadata.to(device)
        targets = targets.float().to(device)

        optimizer.zero_grad()
        outputs = model(images, metadata)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_preds.extend(outputs.detach().cpu().numpy())
        train_targets.extend(targets.cpu().numpy())

    # Validation
    model.eval()
    val_preds = []
    val_targets = []

    with torch.no_grad():
        for images, metadata, targets in val_loader:
            images = images.to(device)
            metadata = metadata.to(device)
            targets = targets.float().to(device)

            outputs = model(images, metadata)
            val_preds.extend(outputs.cpu().numpy())
            val_targets.extend(targets.cpu().numpy())

    # Calculate metrics
    train_auc = roc_auc_score(train_targets, train_preds)
    val_auc = roc_auc_score(val_targets, val_preds)
    avg_train_loss = train_loss / len(train_loader)

    print(f"Epoch {epoch+1}/{num_epochs}:")
    print(
        f"Train Loss: {avg_train_loss:.4f}, Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}"
    )

    # Update scheduler
    scheduler.step(val_auc)

    # Save best model
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        torch.save(model.state_dict(), "./working/best_model.pth")

print(f"\nBest Validation AUC: {best_val_auc:.4f}")

# Prepare test set
test_df, test_meta, _ = preprocess_metadata(test_df, is_train=False)
test_dataset = MelanomaDataset(
    test_df, test_meta, TEST_IMG_PATH, val_transform, is_train=False
)
test_loader = DataLoader(
    test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
)

# Load best model and predict on test set
model.load_state_dict(torch.load("./working/best_model.pth"))
model.eval()

test_preds = []
image_names = []

with torch.no_grad():
    for images, metadata in test_loader:
        images = images.to(device)
        metadata = metadata.to(device)

        outputs = model(images, metadata)
        test_preds.extend(outputs.cpu().numpy())

# Create submission file
submission_df = pd.DataFrame(
    {"image_name": test_df["image_name"], "target": test_preds}
)

# Ensure submission directory exists
os.makedirs("./submission", exist_ok=True)
submission_path = "./submission/submission.csv"
submission_df.to_csv(submission_path, index=False)

print(f"\nSubmission saved to {submission_path}")
print(f"Submission shape: {submission_df.shape}")
print(
    f"Target range: [{submission_df['target'].min():.3f}, {submission_df['target'].max():.3f}]"
)

# Also save validation predictions for reference
val_df = train_df.iloc[val_idx].copy()
val_df["prediction"] = val_preds
val_df[["image_name", "target", "prediction"]].to_csv(
    "./working/validation_predictions.csv", index=False
)
print(f"\nValidation AUC: {best_val_auc:.4f}")
