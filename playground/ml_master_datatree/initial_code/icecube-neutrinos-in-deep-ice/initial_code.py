import numpy as np
import pandas as pd
import os
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# Configuration
DATA_DIR = Path("./input")
SUBMISSION_DIR = Path("./submission")
SUBMISSION_DIR.mkdir(exist_ok=True)

print("Loading sensor geometry...")
sensor_geometry = pd.read_csv(DATA_DIR / "sensor_geometry.csv")
sensor_coords = sensor_geometry[["x", "y", "z"]].values.astype(np.float32)


# Convert coordinates to spherical for reference
def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    zenith = np.arccos(z / r)
    azimuth = np.arctan2(y, x)
    azimuth = np.where(azimuth < 0, azimuth + 2 * np.pi, azimuth)
    return azimuth, zenith


sensor_azimuth, sensor_zenith = cartesian_to_spherical(
    sensor_coords[:, 0], sensor_coords[:, 1], sensor_coords[:, 2]
)


# Simple feature extraction per event
def extract_simple_features(event_pulses):
    """Extract simple aggregated features for fast prediction"""
    # Group by sensor
    sensor_stats = (
        event_pulses.groupby("sensor_id")
        .agg(
            {
                "charge": ["sum", "mean", "count"],
                "time": ["mean", "std"],
                "auxiliary": "mean",
            }
        )
        .fillna(0)
    )

    # Flatten column names
    sensor_stats.columns = [
        "_".join(col).strip() for col in sensor_stats.columns.values
    ]

    return sensor_stats


# Process test data in batches
print("Loading test metadata...")
test_meta = pd.read_parquet(DATA_DIR / "test_meta.parquet")
print(f"Test events: {len(test_meta):,}")

# Get unique batch IDs
test_batch_ids = test_meta["batch_id"].unique()
print(f"Test batches: {len(test_batch_ids)}")

# Initialize predictions
predictions = []
event_ids = []

# Process each batch
for batch_id in test_batch_ids:
    print(f"Processing batch {batch_id}...")

    # Load batch data
    batch_file = DATA_DIR / "test" / f"batch_{int(batch_id)}.parquet"
    batch_data = pd.read_parquet(batch_file)

    # Get events in this batch
    batch_events = test_meta[test_meta["batch_id"] == batch_id]

    # Process each event
    for _, row in batch_events.iterrows():
        event_id = row["event_id"]

        # Get pulses for this event
        event_pulses = batch_data.loc[event_id]  # event_id is the index

        # If event_pulses is a Series (single pulse), convert to DataFrame
        if isinstance(event_pulses, pd.Series):
            event_pulses = pd.DataFrame([event_pulses])

        # Reset index to get sensor_id as column
        event_pulses = event_pulses.reset_index()

        # Simple heuristic: use weighted average of sensor directions by charge
        if len(event_pulses) > 0:
            # Calculate weights based on charge
            weights = event_pulses["charge"].values

            # Get sensor indices
            sensor_indices = event_pulses["sensor_id"].values

            # Weighted average of sensor directions
            valid_mask = sensor_indices < len(sensor_azimuth)
            if valid_mask.any():
                sensor_indices = sensor_indices[valid_mask]
                weights = weights[valid_mask]

                # Get sensor directions
                event_azimuths = sensor_azimuth[sensor_indices]
                event_zeniths = sensor_zenith[sensor_indices]

                # Handle circular mean for azimuth
                sin_sum = np.sum(weights * np.sin(event_azimuths))
                cos_sum = np.sum(weights * np.cos(event_azimuths))
                pred_azimuth = np.arctan2(sin_sum, cos_sum)
                if pred_azimuth < 0:
                    pred_azimuth += 2 * np.pi

                # Linear mean for zenith
                pred_zenith = np.average(event_zeniths, weights=weights)
                pred_zenith = np.clip(pred_zenith, 0, np.pi)
            else:
                pred_azimuth, pred_zenith = np.pi, np.pi / 2
        else:
            pred_azimuth, pred_zenith = np.pi, np.pi / 2

        predictions.append((pred_azimuth, pred_zenith))
        event_ids.append(event_id)

# Create submission dataframe
submission_df = pd.DataFrame(
    {
        "event_id": event_ids,
        "azimuth": [p[0] for p in predictions],
        "zenith": [p[1] for p in predictions],
    }
)

# Sort by event_id to match expected order
submission_df = submission_df.sort_values("event_id")

# Save submission
submission_path = SUBMISSION_DIR / "submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"\nSubmission saved to {submission_path}")
print(f"Number of predictions: {len(submission_df):,}")

# Create a simple validation using a small sample from training data
print("\nCreating validation metric on small training sample...")

# Load small sample of training data for validation
train_meta_sample = pd.read_parquet(DATA_DIR / "train_meta.parquet").sample(
    n=1000, random_state=42
)
train_meta_sample = train_meta_sample.sort_values("batch_id")

val_predictions = []
val_targets = []

# Process a few training batches for validation
processed_batches = set()
for _, row in train_meta_sample.iterrows():
    batch_id = row["batch_id"]

    if batch_id not in processed_batches:
        batch_file = DATA_DIR / "train" / f"batch_{int(batch_id)}.parquet"
        if os.path.exists(batch_file):
            batch_data = pd.read_parquet(batch_file)
            processed_batches.add(batch_id)

    event_id = row["event_id"]

    # Get true values
    true_azimuth = row["azimuth"]
    true_zenith = row["zenith"]
    val_targets.append((true_azimuth, true_zenith))

    # Get pulses for this event
    if event_id in batch_data.index:
        event_pulses = batch_data.loc[event_id]

        if isinstance(event_pulses, pd.Series):
            event_pulses = pd.DataFrame([event_pulses])

        event_pulses = event_pulses.reset_index()

        # Same prediction logic as above
        if len(event_pulses) > 0:
            weights = event_pulses["charge"].values
            sensor_indices = event_pulses["sensor_id"].values

            valid_mask = sensor_indices < len(sensor_azimuth)
            if valid_mask.any():
                sensor_indices = sensor_indices[valid_mask]
                weights = weights[valid_mask]

                event_azimuths = sensor_azimuth[sensor_indices]
                event_zeniths = sensor_zenith[sensor_indices]

                sin_sum = np.sum(weights * np.sin(event_azimuths))
                cos_sum = np.sum(weights * np.cos(event_azimuths))
                pred_azimuth = np.arctan2(sin_sum, cos_sum)
                if pred_azimuth < 0:
                    pred_azimuth += 2 * np.pi

                pred_zenith = np.average(event_zeniths, weights=weights)
                pred_zenith = np.clip(pred_zenith, 0, np.pi)
            else:
                pred_azimuth, pred_zenith = np.pi, np.pi / 2
        else:
            pred_azimuth, pred_zenith = np.pi, np.pi / 2
    else:
        pred_azimuth, pred_zenith = np.pi, np.pi / 2

    val_predictions.append((pred_azimuth, pred_zenith))


# Calculate angular distance metric
def angular_dist_score(az_true, zen_true, az_pred, zen_pred):
    """Calculate mean angular error"""
    # Convert to unit vectors
    true_vec = np.array(
        [
            np.cos(az_true) * np.sin(zen_true),
            np.sin(az_true) * np.sin(zen_true),
            np.cos(zen_true),
        ]
    ).T

    pred_vec = np.array(
        [
            np.cos(az_pred) * np.sin(zen_pred),
            np.sin(az_pred) * np.sin(zen_pred),
            np.cos(zen_pred),
        ]
    ).T

    # Calculate dot product and clip
    dot = np.sum(true_vec * pred_vec, axis=1)
    dot = np.clip(dot, -1, 1)

    # Angular distance
    angular_dist = np.arccos(dot)
    return np.mean(angular_dist)


# Convert to arrays
az_true = np.array([t[0] for t in val_targets])
zen_true = np.array([t[1] for t in val_targets])
az_pred = np.array([p[0] for p in val_predictions])
zen_pred = np.array([p[1] for p in val_predictions])

val_score = angular_dist_score(az_true, zen_true, az_pred, zen_pred)
print(f"Validation Angular Error (on 1000 samples): {val_score:.6f}")

# Verify submission format
print("\nSubmission format check:")
print(f"Columns: {list(submission_df.columns)}")
print(f"Shape: {submission_df.shape}")
print(
    f"Azimuth range: [{submission_df['azimuth'].min():.3f}, {submission_df['azimuth'].max():.3f}]"
)
print(
    f"Zenith range: [{submission_df['zenith'].min():.3f}, {submission_df['zenith'].max():.3f}]"
)
print(f"Null values: {submission_df.isnull().sum().sum()}")

print(f"\n✓ Submission file created successfully!")
