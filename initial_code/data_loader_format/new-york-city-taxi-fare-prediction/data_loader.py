import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import os
import warnings

warnings.filterwarnings("ignore")
np.random.seed(42)


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate haversine distance between two points."""
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def bearing(lat1, lon1, lat2, lon2):
    """Calculate bearing between two points."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return np.degrees(np.arctan2(x, y))


def is_airport(lat, lon, radius_km=2):
    """Check if location is near an airport."""
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
    """Create advanced features for taxi fare prediction."""
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
    """Clean and filter data to NYC area."""
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
    """Load data in chunks to handle large files."""
    chunks = []
    total_rows = 0
    for chunk in pd.read_csv(filepath, chunksize=chunksize):
        chunks.append(chunk)
        total_rows += len(chunk)
        if max_rows and total_rows >= max_rows:
            break
    return pd.concat(chunks, ignore_index=True)


class MyDataLoader(BaseDataLoader):
    """Data loader for NYC Taxi Fare Prediction."""

    def __init__(self, max_rows=5000000, **kwargs):
        super().__init__(**kwargs)
        self.max_rows = max_rows
        self.cluster_model = None
        self.cat_features = [
            "pickup_cluster",
            "dropoff_cluster",
            "hour",
            "day_of_week",
            "month",
            "year",
        ]

    def setup(self):
        """Load data, perform feature engineering, and prepare train/val/test sets."""
        print("Loading and processing data...")

        # Load all training data
        train_full = load_data_in_chunks("./input/labels.csv", max_rows=self.max_rows)
        print(f"Training data loaded: {len(train_full)} rows")

        # Check for pre-defined validation set
        if os.path.exists('input/val.csv'):
            print("Using pre-defined validation set from input/val.csv")
            val_df = pd.read_csv('input/val.csv')
            val_keys = set(val_df['key'].values)

            # Remove validation samples from training data
            train = train_full[~train_full['key'].isin(val_keys)].copy()
            print(f"Training data after removing validation samples: {len(train)} rows")

            # Clean both datasets
            train = clean_data(train, is_train=True)
            val_cleaned = clean_data(val_df, is_train=False)
            print(f"After cleaning - Train: {len(train)} rows, Val: {len(val_cleaned)} rows")

            # Create features for training data (fit clusters)
            print("Creating features with clustering...")
            X_train, self.cluster_model = create_advanced_features(
                train.drop(columns=["fare_amount"]), fit_clusters=True
            )
            y_train = train["fare_amount"].values

            # Create features for validation data
            X_val, _ = create_advanced_features(
                val_cleaned.drop(columns=["fare_amount"], errors='ignore'),
                cluster_model=self.cluster_model,
                fit_clusters=False
            )
            y_val = val_cleaned['fare_amount'].values if 'fare_amount' in val_cleaned.columns else None
        else:
            # No validation file, use time-based split
            print("No validation file found, using time-based split")
            train = clean_data(train_full, is_train=True)
            print(f"After cleaning: {len(train)} rows")

            # Create features with clustering
            print("Creating features with clustering...")
            X, self.cluster_model = create_advanced_features(
                train.drop(columns=["fare_amount"]), fit_clusters=True
            )
            y = train["fare_amount"].values

            # Time-based split (80/20)
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]

        # Remove NaNs from training data
        valid_mask = ~X_train.isna().any(axis=1)
        X_train = X_train[valid_mask]
        y_train = y_train[valid_mask]
        print(f"After removing NaNs from train: {len(X_train)} rows")

        # Handle NaNs in validation data
        if y_val is not None:
            val_valid_mask = ~X_val.isna().any(axis=1)
            X_val = X_val[val_valid_mask]
            y_val = y_val[val_valid_mask]
            print(f"After removing NaNs from val: {len(X_val)} rows")

        # Convert categorical features
        for col in self.cat_features:
            if col in X_train.columns:
                X_train[col] = X_train[col].astype("category")
            if col in X_val.columns:
                X_val[col] = X_val[col].astype("category")

        print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}")

        # Load and process test data
        print("Processing test data...")
        test = pd.read_csv("./input/test.csv")
        test_original = test.copy()

        X_test, _ = create_advanced_features(
            test, cluster_model=self.cluster_model, fit_clusters=False
        )

        # Ensure same columns and handle missing values
        for col in X_train.columns:
            if col not in X_test.columns:
                X_test[col] = 0

        # Align columns
        X_test = X_test[X_train.columns]

        # Handle missing values
        for col in X_test.columns:
            if X_test[col].isna().any():
                if col in self.cat_features:
                    mode_val = X_train[col].mode()[0] if len(X_train[col].mode()) > 0 else 0
                    X_test[col] = X_test[col].fillna(mode_val)
                else:
                    X_test[col] = X_test[col].fillna(0)

        # Convert categorical columns to category type
        for col in self.cat_features:
            if col in X_test.columns:
                X_test[col] = X_test[col].astype("category")

        # Store processed data
        self.train_data = {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'cat_features': self.cat_features
        }
        self.test_data = {
            'X_test': X_test,
            'test_original': test_original
        }

    def describe(self) -> str:
        """Return description of data processing approach."""
        return """
        NYC Taxi Fare Prediction Data Loader
        
        Data Processing:
        - Loads training data from input/labels.csv with chunking support
        - Uses pre-defined validation set from input/val.csv if available
        - Falls back to time-based 80/20 split if no validation file
        
        Data Cleaning:
        - Filters to NYC area coordinates (lat: 40.5-41.0, lon: -74.3 to -73.7)
        - Filters fare_amount to [0, 200] range
        - Filters passenger_count to [1, 6] range
        - Removes NaN values
        
        Feature Engineering:
        - Time features: hour, day_of_week, month, year, day_of_year
        - Fourier features: hour_sin/cos, dow_sin/cos for seasonality
        - Distance features: haversine, manhattan distance
        - Direction features: bearing angle
        - Airport features: pickup_airport, dropoff_airport (JFK, LGA, EWR)
        - NYC center distance features
        - Cluster-based features: MiniBatchKMeans with 50 clusters
        - Speed features: estimated speed_kmh
        
        Categorical Features:
        - pickup_cluster, dropoff_cluster, hour, day_of_week, month, year
        """