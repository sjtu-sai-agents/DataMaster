import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import LabelEncoder
import warnings
import os
from datetime import datetime
import gc
from scipy import stats

warnings.filterwarnings("ignore")
np.random.seed(42)


def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def bearing(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return np.degrees(np.arctan2(x, y))


def is_airport(lat, lon, radius_km=2):
    airports = {
        "JFK": (40.6413, -73.7781),
        "LGA": (40.7769, -73.8740),
        "EWR": (40.6895, -74.1745),
    }
    for airport_lat, airport_lon in airports.values():
        if haversine_distance(lat, lon, airport_lat, airport_lon) <= radius_km:
            return 1
    return 0


def create_advanced_features(df, cluster_model=None, fit_clusters=False):
    df = df.copy()
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])

    # Time features
    df["hour"] = df["pickup_datetime"].dt.hour
    df["day_of_week"] = df["pickup_datetime"].dt.dayofweek
    df["month"] = df["pickup_datetime"].dt.month
    df["year"] = df["pickup_datetime"].dt.year
    df["day_of_year"] = df["pickup_datetime"].dt.dayofyear

    # Fourier features for seasonality
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # Distance features
    df["haversine"] = haversine_distance(
        df["pickup_latitude"],
        df["pickup_longitude"],
        df["dropoff_latitude"],
        df["dropoff_longitude"],
    )
    df["manhattan"] = (
        np.abs(df["dropoff_latitude"] - df["pickup_latitude"])
        + np.abs(df["dropoff_longitude"] - df["pickup_longitude"])
    ) * 111

    # Direction features
    df["bearing"] = bearing(
        df["pickup_latitude"],
        df["pickup_longitude"],
        df["dropoff_latitude"],
        df["dropoff_longitude"],
    )

    # Airport features
    df["pickup_airport"] = df.apply(
        lambda r: is_airport(r["pickup_latitude"], r["pickup_longitude"]), axis=1
    )
    df["dropoff_airport"] = df.apply(
        lambda r: is_airport(r["dropoff_latitude"], r["dropoff_longitude"]), axis=1
    )

    # NYC center features
    nyc_center = (40.7831, -73.9712)
    df["pickup_center_dist"] = haversine_distance(
        df["pickup_latitude"], df["pickup_longitude"], nyc_center[0], nyc_center[1]
    )
    df["dropoff_center_dist"] = haversine_distance(
        df["dropoff_latitude"], df["dropoff_longitude"], nyc_center[0], nyc_center[1]
    )

    # Cluster-based features
    if fit_clusters and cluster_model is None:
        coords = np.vstack(
            [
                df[["pickup_latitude", "pickup_longitude"]].values,
                df[["dropoff_latitude", "dropoff_longitude"]].values,
            ]
        )
        valid_coords = coords[
            (np.abs(coords[:, 0]) < 90) & (np.abs(coords[:, 1]) < 180)
        ]
        cluster_model = MiniBatchKMeans(
            n_clusters=50, random_state=42, batch_size=10000, n_init=3
        )
        cluster_model.fit(valid_coords)

    if cluster_model is not None:
        df["pickup_cluster"] = cluster_model.predict(
            df[["pickup_latitude", "pickup_longitude"]].values
        )
        df["dropoff_cluster"] = cluster_model.predict(
            df[["dropoff_latitude", "dropoff_longitude"]].values
        )

    # Speed features
    df["speed_kmh"] = df["haversine"] / 0.5

    # Remove original columns
    df.drop(columns=["pickup_datetime", "key"], errors="ignore", inplace=True)

    return df, cluster_model


def clean_data(df, is_train=True):
    df = df.copy()

    # Filter to NYC area
    nyc_min_lat, nyc_max_lat = 40.5, 41.0
    nyc_min_lon, nyc_max_lon = -74.3, -73.7

    mask = (
        (df["pickup_latitude"].between(nyc_min_lat, nyc_max_lat))
        & (df["pickup_longitude"].between(nyc_min_lon, nyc_max_lon))
        & (df["dropoff_latitude"].between(nyc_min_lat, nyc_max_lat))
        & (df["dropoff_longitude"].between(nyc_min_lon, nyc_max_lon))
    )
    df = df[mask].copy()

    if is_train:
        df = df[df["fare_amount"].between(0, 200)].copy()
        df = df[df["passenger_count"].between(1, 6)].copy()

    return df


def load_data_in_chunks(filepath, chunksize=1000000, max_rows=None):
    chunks = []
    total_rows = 0
    for chunk in pd.read_csv(filepath, chunksize=chunksize):
        chunks.append(chunk)
        total_rows += len(chunk)
        if max_rows and total_rows >= max_rows:
            break
    return pd.concat(chunks, ignore_index=True)


def main():
    print("Loading and processing data...")

    # Load all training data
    train = load_data_in_chunks("./input/labels.csv", max_rows=5000000)
    print(f"Training data loaded: {len(train)} rows")

    # Clean data
    train = clean_data(train, is_train=True)
    print(f"After cleaning: {len(train)} rows")

    # Create features with clustering
    print("Creating features with clustering...")
    X, cluster_model = create_advanced_features(
        train.drop(columns=["fare_amount"]), fit_clusters=True
    )
    y = train["fare_amount"].values

    # Remove NaNs
    valid_mask = ~X.isna().any(axis=1)
    X = X[valid_mask]
    y = y[valid_mask]
    print(f"After removing NaNs: {len(X)} rows")

    # Time-based split
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    # Convert categorical features
    cat_features = [
        "pickup_cluster",
        "dropoff_cluster",
        "hour",
        "day_of_week",
        "month",
        "year",
    ]
    for col in cat_features:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype("category")
            X_val[col] = X_val[col].astype("category")

    print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}")

    # LightGBM parameters optimized for GPU
    params = {
        "boosting_type": "gbdt",
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.1,
        "num_leaves": 127,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "n_jobs": -1,
        "device": "gpu",
        "gpu_platform_id": 0,
        "gpu_device_id": 0,
        "verbose": -1,
        "seed": 42,
    }

    # Create datasets
    train_data = lgb.Dataset(
        X_train, label=y_train, categorical_feature=cat_features, free_raw_data=False
    )
    val_data = lgb.Dataset(
        X_val, label=y_val, categorical_feature=cat_features, free_raw_data=False
    )

    print("Training LightGBM model...")
    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, val_data],
        valid_names=["train", "val"],
        num_boost_round=1000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=100),
        ],
    )

    # Predict on validation
    y_pred = model.predict(X_val, num_iteration=model.best_iteration)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    print(f"Validation RMSE: {rmse:.4f}")

    # Load and process test data
    print("Processing test data...")
    test = pd.read_csv("./input/test.csv")
    test_original = test.copy()

    X_test, _ = create_advanced_features(
        test, cluster_model=cluster_model, fit_clusters=False
    )

    # Ensure same columns and handle missing values properly
    for col in X_train.columns:
        if col not in X_test.columns:
            X_test[col] = 0

    # Align columns
    X_test = X_test[X_train.columns]

    # Handle missing values: fill numerical columns with 0, categorical columns with mode
    for col in X_test.columns:
        if X_test[col].isna().any():
            if col in cat_features:
                # For categorical columns, fill with the most frequent value from training data
                mode_val = X_train[col].mode()[0] if len(X_train[col].mode()) > 0 else 0
                X_test[col] = X_test[col].fillna(mode_val)
            else:
                # For numerical columns, fill with 0
                X_test[col] = X_test[col].fillna(0)

    # Convert categorical columns to category type
    for col in cat_features:
        if col in X_test.columns:
            X_test[col] = X_test[col].astype("category")

    # Predict on test
    print("Making test predictions...")
    test_pred = model.predict(X_test, num_iteration=model.best_iteration)

    # Post-processing: clip to realistic range
    test_pred = np.clip(test_pred, 2.5, 200)

    # Create submission
    submission = pd.DataFrame({"key": test_original["key"], "fare_amount": test_pred})

    os.makedirs("./submission", exist_ok=True)
    submission_path = "./submission/submission.csv"
    submission.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")

    # Print sample
    print("\nSample predictions:")
    print(submission.head(10))

    return rmse


if __name__ == "__main__":
    final_rmse = main()
    print(f"\nFinal Validation RMSE: {final_rmse:.4f}")
