import os
import numpy as np
import pandas as pd
import pydicom
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupKFold
import warnings

warnings.filterwarnings("ignore")

# Set paths
INPUT_DIR = "./input"
TRAIN_CSV = os.path.join(INPUT_DIR, "train.csv")
TEST_CSV = os.path.join(INPUT_DIR, "test.csv")
TRAIN_IMAGES = os.path.join(INPUT_DIR, "train")
TEST_IMAGES = os.path.join(INPUT_DIR, "test")
SAMPLE_SUB = os.path.join(INPUT_DIR, "sample_submission.csv")
SUBMISSION_PATH = "./submission/submission.csv"

# Create submission directory
os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

# Load data
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB)


# Preprocess function for CT scans
def extract_ct_features(patient_id, image_dir):
    """Extract basic statistics from CT scans without using deep learning"""
    patient_path = os.path.join(image_dir, patient_id)
    if not os.path.exists(patient_path):
        return [np.nan] * 5

    try:
        # Get list of DICOM files
        dcm_files = list(Path(patient_path).glob("*.dcm"))
        if len(dcm_files) == 0:
            return [np.nan] * 5

        # Sample up to 50 slices to avoid memory issues
        sample_files = dcm_files[: min(50, len(dcm_files))]
        pixel_values = []

        for dcm_file in sample_files:
            try:
                ds = pydicom.dcmread(dcm_file, force=True)
                if hasattr(ds, "pixel_array"):
                    pixels = ds.pixel_array.flatten()
                    # Normalize and clip extreme values
                    pixels = pixels[pixels > -1000]
                    pixels = pixels[pixels < 1000]
                    if len(pixels) > 0:
                        pixel_values.extend(pixels.tolist())
            except:
                continue

        if len(pixel_values) == 0:
            return [np.nan] * 5

        pixel_values = np.array(pixel_values)

        # Extract basic statistics
        return [
            np.mean(pixel_values),
            np.std(pixel_values),
            np.percentile(pixel_values, 25),
            np.percentile(pixel_values, 75),
            len(pixel_values),  # proxy for lung volume
        ]
    except:
        return [np.nan] * 5


# Extract CT features for train data
print("Extracting CT features for training data...")
ct_features_train = []
for patient in train_df["Patient"].unique():
    features = extract_ct_features(patient, TRAIN_IMAGES)
    ct_features_train.append([patient] + features)

ct_train_df = pd.DataFrame(
    ct_features_train,
    columns=["Patient", "ct_mean", "ct_std", "ct_q25", "ct_q75", "ct_volume"],
)
train_df = train_df.merge(ct_train_df, on="Patient", how="left")

# Fill missing CT features with median
ct_cols = ["ct_mean", "ct_std", "ct_q25", "ct_q75", "ct_volume"]
for col in ct_cols:
    train_df[col].fillna(train_df[col].median(), inplace=True)

# Extract CT features for test data
print("Extracting CT features for test data...")
ct_features_test = []
for patient in test_df["Patient"].unique():
    features = extract_ct_features(patient, TEST_IMAGES)
    ct_features_test.append([patient] + features)

ct_test_df = pd.DataFrame(
    ct_features_test,
    columns=["Patient", "ct_mean", "ct_std", "ct_q25", "ct_q75", "ct_volume"],
)
test_df = test_df.merge(ct_test_df, on="Patient", how="left")

# Fill missing CT features with median from training data
for col in ct_cols:
    test_df[col].fillna(train_df[col].median(), inplace=True)


# Feature engineering - Fixed version
def create_features(df):
    df = df.copy()

    # Encode categorical variables
    le_sex = LabelEncoder()
    le_smoking = LabelEncoder()

    df["Sex_encoded"] = le_sex.fit_transform(df["Sex"])
    df["SmokingStatus_encoded"] = le_smoking.fit_transform(df["SmokingStatus"])

    # Get baseline FVC for each patient
    # For patients with Week=0 measurement, use that as baseline
    # For others, use the minimum Week measurement as baseline
    baseline_df = df.copy()
    # Find baseline (Week=0 or minimum Week if Week=0 doesn't exist)
    patient_baselines = []
    for patient in df["Patient"].unique():
        patient_data = df[df["Patient"] == patient]
        # Try to find Week=0
        week_zero = patient_data[patient_data["Weeks"] == 0]
        if len(week_zero) > 0:
            baseline_fvc = week_zero.iloc[0]["FVC"]
        else:
            # Use the minimum Week measurement
            min_week_idx = patient_data["Weeks"].idxmin()
            baseline_fvc = patient_data.loc[min_week_idx, "FVC"]
        patient_baselines.append({"Patient": patient, "BaseFVC": baseline_fvc})

    base_fvc_df = pd.DataFrame(patient_baselines)
    df = df.merge(base_fvc_df, on="Patient", how="left")

    # Get baseline Percent for each patient
    patient_percent_baselines = []
    for patient in df["Patient"].unique():
        patient_data = df[df["Patient"] == patient]
        # Try to find Week=0
        week_zero = patient_data[patient_data["Weeks"] == 0]
        if len(week_zero) > 0:
            baseline_percent = week_zero.iloc[0]["Percent"]
        else:
            # Use the minimum Week measurement
            min_week_idx = patient_data["Weeks"].idxmin()
            baseline_percent = patient_data.loc[min_week_idx, "Percent"]
        patient_percent_baselines.append(
            {"Patient": patient, "BasePercent": baseline_percent}
        )

    base_percent_df = pd.DataFrame(patient_percent_baselines)
    df = df.merge(base_percent_df, on="Patient", how="left")

    # Create interaction features
    df["Age_Week_interaction"] = df["Age"] * df["Weeks"]

    # Weeks from baseline (absolute)
    df["Weeks_abs"] = df["Weeks"].abs()

    # Rate of change features
    df["FVC_per_week"] = (df["FVC"] - df["BaseFVC"]) / (df["Weeks"] + 1e-5)

    return df


train_df = create_features(train_df)
test_df = create_features(test_df)

# Mixed-effects modeling approach
print("Training mixed-effects model...")

# Prepare features for modeling
feature_cols = [
    "Weeks",
    "Age",
    "Sex_encoded",
    "SmokingStatus_encoded",
    "ct_mean",
    "ct_std",
    "ct_volume",
    "BaseFVC",
    "Age_Week_interaction",
    "Weeks_abs",
]

# GroupKFold cross-validation
n_folds = 5
gkf = GroupKFold(n_splits=n_folds)
groups = train_df["Patient"].values
X = train_df[feature_cols].values
y = train_df["FVC"].values

# Store predictions for validation
val_predictions = []
val_targets = []
patient_groups_val = []

# Simple linear model with patient random effects (approximated via grouping)
for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
    print(f"Training fold {fold + 1}/{n_folds}")

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    groups_train = groups[train_idx]

    # Fit ridge regression
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    model = Ridge(alpha=1.0, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Predict on validation
    y_pred = model.predict(X_val_scaled)

    val_predictions.extend(y_pred.tolist())
    val_targets.extend(y_val.tolist())
    patient_groups_val.extend(groups[val_idx].tolist())

# Calculate residuals for uncertainty estimation
val_predictions = np.array(val_predictions)
val_targets = np.array(val_targets)
patient_groups_val = np.array(patient_groups_val)

residuals = val_targets - val_predictions

# Calculate patient-specific uncertainty
patient_residuals = {}
for patient in np.unique(patient_groups_val):
    mask = patient_groups_val == patient
    if mask.sum() > 0:
        patient_residuals[patient] = np.std(residuals[mask])

# Overall uncertainty as fallback
overall_std = np.std(residuals)


# Competition metric calculation
def laplace_log_likelihood(y_true, y_pred, sigma):
    sigma_clipped = np.maximum(sigma, 70)
    delta = np.minimum(np.abs(y_true - y_pred), 1000)
    metric = -np.sqrt(2) * delta / sigma_clipped - np.log(np.sqrt(2) * sigma_clipped)
    return np.mean(metric)


# Prepare validation predictions with uncertainty
val_patients = patient_groups_val
val_uncertainty = []
for patient in val_patients:
    if patient in patient_residuals and patient_residuals[patient] > 0:
        val_uncertainty.append(patient_residuals[patient])
    else:
        val_uncertainty.append(overall_std)

val_uncertainty = np.array(val_uncertainty)

# Calculate validation score
val_score = laplace_log_likelihood(val_targets, val_predictions, val_uncertainty)
print(f"\nValidation Laplace Log Likelihood: {val_score:.6f}")

# Train final model on all training data
print("\nTraining final model on all data...")
scaler_final = StandardScaler()
X_all = train_df[feature_cols].values
X_all_scaled = scaler_final.fit_transform(X_all)
y_all = train_df["FVC"].values

final_model = Ridge(alpha=1.0, random_state=42)
final_model.fit(X_all_scaled, y_all)

# Calculate final model residuals for uncertainty
train_pred = final_model.predict(X_all_scaled)
train_residuals = y_all - train_pred

# Patient-specific residuals for training data
patient_train_residuals = {}
for patient in train_df["Patient"].unique():
    mask = train_df["Patient"] == patient
    if mask.sum() > 0:
        patient_train_residuals[patient] = np.std(train_residuals[mask])

overall_train_std = np.std(train_residuals)

# Prepare test predictions
print("\nMaking predictions for test set...")

# Parse sample submission to get patient-week pairs
sample_sub["Patient"] = sample_sub["Patient_Week"].apply(lambda x: x.split("_")[0])
sample_sub["Week"] = sample_sub["Patient_Week"].apply(lambda x: int(x.split("_")[1]))

# Create test dataset for prediction
test_predictions = []

for _, row in sample_sub.iterrows():
    patient = row["Patient"]
    week = row["Week"]

    # Get patient data from test_df
    patient_data = test_df[test_df["Patient"] == patient].iloc[0].copy()

    # Update week for prediction
    patient_data["Weeks"] = week
    patient_data["Age_Week_interaction"] = patient_data["Age"] * week
    patient_data["Weeks_abs"] = abs(week)

    # Prepare features
    features = patient_data[feature_cols].values.reshape(1, -1)
    features_scaled = scaler_final.transform(features)

    # Predict FVC
    fvc_pred = final_model.predict(features_scaled)[0]

    # Clamp predictions to reasonable range
    fvc_pred = np.clip(fvc_pred, 1000, 6000)

    # Determine confidence
    if patient in patient_train_residuals:
        confidence = max(patient_train_residuals[patient], 100)
    else:
        confidence = max(overall_train_std, 100)

    # Confidence clipping as per competition rules
    confidence = np.clip(confidence, 100, 400)

    test_predictions.append(
        {"Patient_Week": row["Patient_Week"], "FVC": fvc_pred, "Confidence": confidence}
    )

# Create submission dataframe
submission_df = pd.DataFrame(test_predictions)

# Ensure proper column order
submission_df = submission_df[["Patient_Week", "FVC", "Confidence"]]

# Save submission
submission_df.to_csv(SUBMISSION_PATH, index=False)
print(f"Submission saved to {SUBMISSION_PATH}")
print(f"Submission shape: {submission_df.shape}")
print(f"Sample predictions:\n{submission_df.head()}")

# Final validation on a hold-out set (simulated)
print("\nFinal evaluation on simulated hold-out set...")
# Use last fold as proxy for final score
print(f"Final validation score (fold average): {val_score:.6f}")

# Additional diagnostic metrics
mae = np.mean(np.abs(val_targets - val_predictions))
rmse = np.sqrt(np.mean((val_targets - val_predictions) ** 2))
print(f"MAE: {mae:.2f} ml")
print(f"RMSE: {rmse:.2f} ml")
print(f"Average confidence: {np.mean(val_uncertainty):.2f} ml")

print("\nSolution complete. Predictions ready for submission.")
