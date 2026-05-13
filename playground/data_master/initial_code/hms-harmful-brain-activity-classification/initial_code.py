import numpy as np
import pandas as pd
import os
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# Set paths
input_dir = Path("./input")
working_dir = Path("./working")
submission_dir = Path("./submission")
submission_dir.mkdir(exist_ok=True)

# Load data
train_df = pd.read_csv(input_dir / "train.csv")
test_df = pd.read_csv(input_dir / "test.csv")
sample_submission = pd.read_csv(input_dir / "sample_submission.csv")

# Define target columns
target_cols = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]


# Function to extract features from EEG data
def extract_eeg_features(eeg_data):
    """Extract statistical features from EEG channels"""
    features = []

    # For each EEG channel (excluding time column if present)
    for col in eeg_data.columns:
        if col in [
            "Fp1",
            "Fp2",
            "F3",
            "F4",
            "C3",
            "C4",
            "P3",
            "P4",
            "O1",
            "O2",
            "F7",
            "F8",
            "T3",
            "T4",
            "T5",
            "T6",
            "Fz",
            "Cz",
            "Pz",
            "EKG",
        ]:
            channel_data = eeg_data[col].values

            # Basic statistical features
            features.extend(
                [
                    np.mean(channel_data),
                    np.std(channel_data),
                    np.min(channel_data),
                    np.max(channel_data),
                    np.median(channel_data),
                    np.percentile(channel_data, 25),
                    np.percentile(channel_data, 75),
                    np.mean(np.abs(channel_data)),
                    np.mean(np.diff(channel_data)),
                    np.std(np.diff(channel_data)),
                ]
            )

    return np.array(features)


# Process training data
print("Processing training data...")
X_train = []
y_train = []

# For efficiency, process a subset (adjust based on time constraints)
sample_indices = np.random.choice(
    len(train_df), min(20000, len(train_df)), replace=False
)

for idx in sample_indices:
    row = train_df.iloc[idx]
    eeg_id = row["eeg_id"]

    try:
        # Load EEG data
        eeg_path = input_dir / "train_eegs" / f"{eeg_id}.parquet"
        if eeg_path.exists():
            eeg_data = pd.read_parquet(eeg_path)

            # Extract features
            features = extract_eeg_features(eeg_data)
            X_train.append(features)

            # Get target probabilities (normalize votes)
            votes = row[target_cols].values.astype(float)
            probs = votes / votes.sum()
            y_train.append(probs)
    except:
        continue

X_train = np.array(X_train)
y_train = np.array(y_train)

# Process test data
print("Processing test data...")
X_test = []
eeg_ids = []

for _, row in test_df.iterrows():
    eeg_id = row["eeg_id"]

    try:
        eeg_path = input_dir / "test_eegs" / f"{eeg_id}.parquet"
        if eeg_path.exists():
            eeg_data = pd.read_parquet(eeg_path)
            features = extract_eeg_features(eeg_data)
            X_test.append(features)
            eeg_ids.append(eeg_id)
    except:
        # If file doesn't exist, use mean features
        X_test.append(np.zeros(X_train.shape[1]) if len(X_train) > 0 else np.zeros(200))
        eeg_ids.append(eeg_id)

X_test = np.array(X_test)

# Handle case where no training data was processed
if len(X_train) == 0:
    print("No training data processed. Using default predictions.")
    submission = sample_submission.copy()
    submission.to_csv(submission_dir / "submission.csv", index=False)
    print("Created default submission file")
    exit()

# Split into train/validation
from sklearn.model_selection import train_test_split

X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# Train XGBoost models for each class (multi-output regression)
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor

print("Training model...")
model = MultiOutputRegressor(
    xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )
)

model.fit(X_train_split, y_train_split)

# Predict on validation set
y_val_pred = model.predict(X_val_split)

# Ensure predictions are non-negative and normalized
y_val_pred = np.clip(y_val_pred, 1e-10, 1)
y_val_pred = y_val_pred / y_val_pred.sum(axis=1, keepdims=True)


# Calculate KL divergence as evaluation metric
def kl_divergence(y_true, y_pred):
    """Calculate KL divergence between true and predicted distributions"""
    # Add small epsilon to avoid log(0)
    eps = 1e-10
    y_true = np.clip(y_true, eps, 1)
    y_pred = np.clip(y_pred, eps, 1)

    return np.mean(np.sum(y_true * np.log(y_true / y_pred), axis=1))


val_kl = kl_divergence(y_val_split, y_val_pred)
print(f"Validation KL Divergence: {val_kl:.6f}")

# Retrain on full training data
print("Retraining on full training data...")
full_model = MultiOutputRegressor(
    xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )
)
full_model.fit(X_train, y_train)

# Predict on test set
print("Making test predictions...")
test_preds = full_model.predict(X_test)

# Ensure proper formatting
test_preds = np.clip(test_preds, 1e-10, 1)
test_preds = test_preds / test_preds.sum(axis=1, keepdims=True)

# Create submission dataframe
submission_df = pd.DataFrame(
    {
        "eeg_id": eeg_ids,
        "seizure_vote": test_preds[:, 0],
        "lpd_vote": test_preds[:, 1],
        "gpd_vote": test_preds[:, 2],
        "lrda_vote": test_preds[:, 3],
        "grda_vote": test_preds[:, 4],
        "other_vote": test_preds[:, 5],
    }
)

# Ensure all test IDs are included (fill missing with uniform probabilities)
all_eeg_ids = test_df["eeg_id"].values
missing_ids = set(all_eeg_ids) - set(eeg_ids)

if missing_ids:
    print(
        f"Warning: {len(missing_ids)} test IDs not processed. Using uniform predictions."
    )
    missing_df = pd.DataFrame(
        {
            "eeg_id": list(missing_ids),
            "seizure_vote": 1 / 6,
            "lpd_vote": 1 / 6,
            "gpd_vote": 1 / 6,
            "lrda_vote": 1 / 6,
            "grda_vote": 1 / 6,
            "other_vote": 1 / 6,
        }
    )
    submission_df = pd.concat([submission_df, missing_df], ignore_index=True)

# Sort by eeg_id to match sample submission order
submission_df = submission_df.sort_values("eeg_id").reset_index(drop=True)

# Verify probabilities sum to 1
row_sums = submission_df[target_cols].sum(axis=1)
if not np.allclose(row_sums, 1.0, atol=1e-10):
    print(
        f"Normalizing rows to sum to 1 (min sum: {row_sums.min():.6f}, max sum: {row_sums.max():.6f})"
    )
    submission_df[target_cols] = submission_df[target_cols].div(row_sums, axis=0)

# Save submission
submission_df.to_csv(submission_dir / "submission.csv", index=False)
print(f"Submission saved to {submission_dir / 'submission.csv'}")
print(f"Submission shape: {submission_df.shape}")
print(f"First few predictions:")
print(submission_df.head())

# Validate submission format matches sample
print("\nValidating submission format...")
print(f"Columns match: {set(submission_df.columns) == set(sample_submission.columns)}")
print(f"Number of rows: {len(submission_df)} (expected: {len(sample_submission)})")
print(
    f"All probabilities sum to 1: {np.allclose(submission_df[target_cols].sum(axis=1), 1.0)}"
)
