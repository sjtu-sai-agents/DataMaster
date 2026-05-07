import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
import os
import gc

# Load data
train = pd.read_csv("./input/train.csv")
test = pd.read_csv("./input/test.csv")


# Feature engineering function
def add_features(df):
    df = df.copy()
    # Basic interactions
    df["R_x_C"] = df["R"] * df["C"]
    df["R_x_u_in"] = df["R"] * df["u_in"]
    df["C_x_u_in"] = df["C"] * df["u_in"]
    df["R_x_time"] = df["R"] * df["time_step"]
    df["C_x_time"] = df["C"] * df["time_step"]

    # Lag features for u_in within each breath
    for lag in [1, 2, 3, 4, 5]:
        df[f"u_in_lag{lag}"] = df.groupby("breath_id")["u_in"].shift(lag)

    # Rolling statistics within each breath
    windows = [3, 5, 10]
    for window in windows:
        df[f"u_in_rolling_mean_{window}"] = (
            df.groupby("breath_id")["u_in"]
            .rolling(window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        df[f"u_in_rolling_std_{window}"] = (
            df.groupby("breath_id")["u_in"]
            .rolling(window, min_periods=1)
            .std()
            .reset_index(level=0, drop=True)
        )
        df[f"u_in_rolling_min_{window}"] = (
            df.groupby("breath_id")["u_in"]
            .rolling(window, min_periods=1)
            .min()
            .reset_index(level=0, drop=True)
        )
        df[f"u_in_rolling_max_{window}"] = (
            df.groupby("breath_id")["u_in"]
            .rolling(window, min_periods=1)
            .max()
            .reset_index(level=0, drop=True)
        )

    # Exponential moving average
    df["u_in_ewm"] = (
        df.groupby("breath_id")["u_in"]
        .ewm(alpha=0.3)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # Cumulative features
    df["u_in_cumsum"] = df.groupby("breath_id")["u_in"].cumsum()
    df["u_in_cumsum_insp"] = (
        df.groupby("breath_id")
        .apply(lambda x: x["u_in"].where(x["u_out"] == 0, 0).cumsum())
        .reset_index(level=0, drop=True)
    )
    df["time_since_start"] = df.groupby("breath_id")["time_step"].transform(
        lambda x: x - x.min()
    )

    # Breath-wise statistics
    df["breath_u_in_mean"] = df.groupby("breath_id")["u_in"].transform("mean")
    df["breath_u_in_std"] = df.groupby("breath_id")["u_in"].transform("std")
    df["breath_time_max"] = df.groupby("breath_id")["time_step"].transform("max")

    # Interaction with u_out
    df["u_in_x_u_out"] = df["u_in"] * df["u_out"]
    df["R_x_u_out"] = df["R"] * df["u_out"]
    df["C_x_u_out"] = df["C"] * df["u_out"]

    # Polynomial features
    df["u_in_squared"] = df["u_in"] ** 2
    df["time_step_squared"] = df["time_step"] ** 2

    # Fill missing values
    df.fillna(method="bfill", inplace=True)
    return df


print("Engineering features...")
train_feats = add_features(train)
test_feats = add_features(test)

# Feature columns
feature_cols = [
    col for col in train_feats.columns if col not in ["id", "breath_id", "pressure"]
]

# Split by breath_id for validation (last 20% of breaths)
breath_ids = train_feats["breath_id"].unique()
val_breath_ids = breath_ids[-int(0.2 * len(breath_ids)) :]
train_mask = ~train_feats["breath_id"].isin(val_breath_ids)
val_mask = train_feats["breath_id"].isin(val_breath_ids)

X_train = train_feats.loc[train_mask, feature_cols]
y_train = train_feats.loc[train_mask, "pressure"]
X_val = train_feats.loc[val_mask, feature_cols]
y_val = train_feats.loc[val_mask, "pressure"]
u_out_val = train_feats.loc[val_mask, "u_out"]

print(f"Training on {len(X_train)} samples, validating on {len(X_val)} samples.")

# LightGBM parameters
params = {
    "objective": "regression_l1",
    "metric": "mae",
    "boosting": "gbdt",
    "device": "gpu",
    "gpu_platform_id": 0,
    "gpu_device_id": 0,
    "num_leaves": 127,
    "max_depth": -1,
    "learning_rate": 0.1,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "seed": 42,
    "n_jobs": -1,
}

# Train with early stopping
train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

model = lgb.train(
    params,
    train_data,
    valid_sets=[val_data],
    num_boost_round=10000,
    callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(100)],
)

# Predict on validation set
val_pred = model.predict(X_val, num_iteration=model.best_iteration)
insp_mask = u_out_val == 0
val_mae = mean_absolute_error(y_val[insp_mask], val_pred[insp_mask])
print(f"Validation MAE (inspiratory phases): {val_mae:.6f}")

# Retrain on full training data (without validation split) for final submission
print("Retraining on full training set...")
X_full = train_feats[feature_cols]
y_full = train_feats["pressure"]

full_data = lgb.Dataset(X_full, label=y_full)
final_model = lgb.train(params, full_data, num_boost_round=model.best_iteration)

# Predict on test set
test_pred = final_model.predict(test_feats[feature_cols])

# Prepare submission
os.makedirs("./submission", exist_ok=True)
submission = pd.DataFrame({"id": test["id"], "pressure": test_pred})
submission_path = "./submission/submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")

# Clean up memory
del train, test, train_feats, test_feats, X_train, X_val, X_full
gc.collect()
