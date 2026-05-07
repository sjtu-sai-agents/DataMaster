import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import pydicom
from tqdm import tqdm
import xgboost as xgb
import warnings
import cv2

warnings.filterwarnings("ignore")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load data
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

# Create prediction_id for train (patient_id + laterality)
train_df["prediction_id"] = (
    train_df["patient_id"].astype(str) + "-" + train_df["laterality"]
)


# Filter train_df to only include rows where images exist
def filter_existing_images(df, image_dir):
    existing_rows = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Checking image existence"):
        patient_id = str(row["patient_id"])
        image_id = str(row["image_id"])
        img_path = os.path.join(image_dir, patient_id, f"{image_id}.dcm")
        if os.path.exists(img_path):
            existing_rows.append(idx)
    return df.loc[existing_rows].reset_index(drop=True)


print("Filtering training data...")
train_df = filter_existing_images(train_df, "./input/train_images")
print(f"Training samples after filtering: {len(train_df)}")

print("Filtering test data...")
test_df = filter_existing_images(test_df, "./input/test_images")
print(f"Test samples after filtering: {len(test_df)}")

# Prepare tabular features
tabular_features = [
    "age",
    "laterality",
    "view",
    "implant",
    "density",
    "machine_id",
    "site_id",
]
categorical_features = [
    "laterality",
    "view",
    "implant",
    "density",
    "machine_id",
    "site_id",
]


# Preprocess tabular data
def preprocess_tabular(df, train=True, imputers=None, encoders=None, scaler=None):
    df = df.copy()

    # Create features
    for col in tabular_features:
        if col not in df.columns:
            df[col] = np.nan

    # Handle categorical features
    if train:
        imputers = {}
        encoders = {}
        for col in categorical_features:
            # Impute missing values with mode
            imputer = SimpleImputer(strategy="most_frequent")
            df[col] = imputer.fit_transform(df[[col]]).ravel()
            imputers[col] = imputer

            # Label encode
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col])
            encoders[col] = encoder

        # Handle age (continuous)
        age_imputer = SimpleImputer(strategy="median")
        df["age"] = age_imputer.fit_transform(df[["age"]]).ravel()
        imputers["age"] = age_imputer

        # Scale numerical features
        scaler = StandardScaler()
        numerical_cols = ["age", "machine_id", "site_id"]
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    else:
        for col in categorical_features:
            if col in imputers:
                df[col] = imputers[col].transform(df[[col]]).ravel()
            if col in encoders:
                # Handle unseen categories
                mask = df[col].isin(encoders[col].classes_)
                if not mask.all():
                    df.loc[~mask, col] = encoders[col].classes_[0]
                df[col] = encoders[col].transform(df[col])

        if "age" in imputers:
            df["age"] = imputers["age"].transform(df[["age"]]).ravel()

        if scaler:
            numerical_cols = ["age", "machine_id", "site_id"]
            df[numerical_cols] = scaler.transform(df[numerical_cols])

    return df[tabular_features], imputers, encoders, scaler


# Simple CNN for image feature extraction
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, 256)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class MammogramDataset(Dataset):
    def __init__(self, df, image_dir, transform=None, max_samples=None):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.max_samples = max_samples

    def __len__(self):
        if self.max_samples:
            return min(self.max_samples, len(self.df))
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = str(row["patient_id"])
        image_id = str(row["image_id"])

        # Load DICOM
        img_path = os.path.join(self.image_dir, patient_id, f"{image_id}.dcm")
        try:
            dicom = pydicom.dcmread(img_path)
            img = dicom.pixel_array

            # Normalize to [0, 1]
            img = img.astype(np.float32)
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)

            # Resize to 256x256
            img = cv2.resize(img, (256, 256))

            # Add channel dimension
            img = np.expand_dims(img, axis=0)

            return torch.FloatTensor(img), row["prediction_id"]
        except Exception as e:
            # Return zero image if loading fails
            zero_img = np.zeros((1, 256, 256), dtype=np.float32)
            return torch.FloatTensor(zero_img), row["prediction_id"]


def extract_image_features(df, image_dir, batch_size=32, max_samples=None):
    """Extract features using simple CNN"""
    model = SimpleCNN().to(device)
    model.eval()

    # Dataset and dataloader
    dataset = MammogramDataset(df, image_dir, max_samples=max_samples)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    # Extract features
    features_dict = {}
    with torch.no_grad():
        for batch, pred_ids in tqdm(dataloader, desc="Extracting image features"):
            batch = batch.to(device)
            features = model(batch).cpu().numpy()

            for i, pred_id in enumerate(pred_ids):
                if pred_id not in features_dict:
                    features_dict[pred_id] = []
                features_dict[pred_id].append(features[i])

    # Average features per prediction_id
    image_features = {}
    for pred_id, feat_list in features_dict.items():
        image_features[pred_id] = np.mean(feat_list, axis=0)

    return image_features


# Probabilistic F1 score implementation
def probabilistic_f1_score(y_true, y_pred, beta=1, eps=1e-7):
    """Calculate probabilistic F1 score"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    pTP = np.sum(y_pred * y_true)
    pFP = np.sum(y_pred * (1 - y_true))
    FN = np.sum((1 - y_pred) * y_true)

    pPrecision = pTP / (pTP + pFP + eps)
    pRecall = pTP / (pTP + FN + eps)

    f1 = (1 + beta**2) * (pPrecision * pRecall) / (beta**2 * pPrecision + pRecall + eps)
    return f1


# Main execution
print("Preprocessing tabular data...")
train_tabular, imputers, encoders, scaler = preprocess_tabular(train_df, train=True)
test_tabular, _, _, _ = preprocess_tabular(
    test_df, train=False, imputers=imputers, encoders=encoders, scaler=scaler
)

# Add prediction_id to tabular data for merging
train_tabular["prediction_id"] = train_df["prediction_id"].values
test_tabular["prediction_id"] = test_df["prediction_id"].values

# Group train data by prediction_id for aggregation
train_agg = (
    train_df.groupby("prediction_id")
    .agg({"cancer": "first", "patient_id": "first"})
    .reset_index()
)

# Extract image features - use smaller sample for speed
print("Extracting image features from training set...")
# Limit to 1000 images for speed
train_image_features = extract_image_features(
    train_df, "./input/train_images", batch_size=32, max_samples=1000
)

print("Extracting image features from test set...")
# Limit to 500 images for speed
test_image_features = extract_image_features(
    test_df, "./input/test_images", batch_size=32, max_samples=500
)


# Combine features
def create_feature_matrix(df_agg, tabular_df, image_features):
    """Create combined feature matrix for prediction_ids"""
    features = []
    pred_ids = []
    targets = []
    patient_ids = []

    for _, row in df_agg.iterrows():
        pred_id = row["prediction_id"]

        if pred_id in image_features:
            # Get tabular features for this prediction_id
            tab_rows = tabular_df[tabular_df["prediction_id"] == pred_id]
            if len(tab_rows) > 0:
                # Take the first row
                tab_row = tab_rows.iloc[0][tabular_features].values.astype(np.float32)

                # Combine with image features
                img_feat = image_features[pred_id].astype(np.float32)
                combined = np.concatenate([tab_row, img_feat])

                features.append(combined)
                pred_ids.append(pred_id)
                patient_ids.append(row["patient_id"])
                if "cancer" in row:
                    targets.append(row["cancer"])

    if len(targets) > 0:
        return np.array(features), np.array(targets), pred_ids, patient_ids
    else:
        return np.array(features), pred_ids, patient_ids


print("Creating combined features...")
X_train, y_train, train_pred_ids, train_patient_ids = create_feature_matrix(
    train_agg, train_tabular, train_image_features
)

# Create test features
test_agg = test_df.groupby("prediction_id").agg({"patient_id": "first"}).reset_index()
X_test, test_pred_ids, _ = create_feature_matrix(
    test_agg, test_tabular, test_image_features
)

# Split by patient for validation - FIX: Use patient_ids at prediction_id level
gkf = GroupKFold(n_splits=5)
train_idx, val_idx = next(gkf.split(X_train, y_train, groups=train_patient_ids))

X_train_fold, X_val = X_train[train_idx], X_train[val_idx]
y_train_fold, y_val = y_train[train_idx], y_train[val_idx]

# Train XGBoost
print("Training XGBoost model...")
dtrain = xgb.DMatrix(X_train_fold, label=y_train_fold)
dval = xgb.DMatrix(X_val, label=y_val)

params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 6,
    "learning_rate": 0.01,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "seed": 42,
    "tree_method": "gpu_hist" if torch.cuda.is_available() else "hist",
}

evals = [(dtrain, "train"), (dval, "eval")]
model = xgb.train(
    params,
    dtrain,
    num_boost_round=500,
    evals=evals,
    early_stopping_rounds=50,
    verbose_eval=False,
)

# Predict on validation set
val_preds = model.predict(dval)
val_f1 = probabilistic_f1_score(y_val, val_preds)
print(f"Validation probabilistic F1 score: {val_f1:.4f}")

# Predict on test set
print("Making test predictions...")
dtest = xgb.DMatrix(X_test)
test_preds = model.predict(dtest)

# Create submission file
submission_df = pd.DataFrame({"prediction_id": test_pred_ids, "cancer": test_preds})

# Ensure all test prediction_ids are included
all_test_ids = test_df["prediction_id"].unique()
missing_ids = set(all_test_ids) - set(test_pred_ids)
if missing_ids:
    print(
        f"Warning: {len(missing_ids)} prediction_ids missing from features. Filling with 0.5"
    )
    missing_df = pd.DataFrame({"prediction_id": list(missing_ids), "cancer": 0.5})
    submission_df = pd.concat([submission_df, missing_df], ignore_index=True)

# Save submission
os.makedirs("./submission", exist_ok=True)
submission_df.to_csv("./submission/submission.csv", index=False)
print(f"Submission saved to ./submission/submission.csv")
print(f"Submission shape: {submission_df.shape}")

# Print some example predictions
print("\nSample predictions:")
print(submission_df.head(10))
print(f"\nValidation pF1 score: {val_f1:.6f}")
