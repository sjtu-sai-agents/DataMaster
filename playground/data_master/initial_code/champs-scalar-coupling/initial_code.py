import numpy as np
import pandas as pd
import os
from pathlib import Path
from sklearn.model_selection import KFold, GroupKFold
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore")

# Set paths
INPUT_PATH = Path("./input")
SUBMISSION_PATH = Path("./submission")
WORKING_PATH = Path("./working")
SUBMISSION_PATH.mkdir(exist_ok=True)
WORKING_PATH.mkdir(exist_ok=True)

# Load data
print("Loading data...")
train = pd.read_csv(INPUT_PATH / "train.csv")
test = pd.read_csv(INPUT_PATH / "test.csv")
structures = pd.read_csv(INPUT_PATH / "structures.csv")
dipole = pd.read_csv(INPUT_PATH / "dipole_moments.csv")
magnetic = pd.read_csv(INPUT_PATH / "magnetic_shielding_tensors.csv")
mulliken = pd.read_csv(INPUT_PATH / "mulliken_charges.csv")
potential = pd.read_csv(INPUT_PATH / "potential_energy.csv")
contributions = pd.read_csv(INPUT_PATH / "scalar_coupling_contributions.csv")


# Merge structure data for atom0 and atom1
def merge_structure_data(df, structures):
    df = df.copy()
    # Get coordinates for atom_index_0
    df = pd.merge(
        df,
        structures,
        how="left",
        left_on=["molecule_name", "atom_index_0"],
        right_on=["molecule_name", "atom_index"],
    )
    df = df.rename(columns={"atom": "atom_0", "x": "x_0", "y": "y_0", "z": "z_0"})
    df = df.drop("atom_index", axis=1)

    # Get coordinates for atom_index_1
    df = pd.merge(
        df,
        structures,
        how="left",
        left_on=["molecule_name", "atom_index_1"],
        right_on=["molecule_name", "atom_index"],
    )
    df = df.rename(columns={"atom": "atom_1", "x": "x_1", "y": "y_1", "z": "z_1"})
    df = df.drop("atom_index", axis=1)

    return df


print("Merging structure data...")
train = merge_structure_data(train, structures)
test = merge_structure_data(test, structures)


# Calculate basic geometric features
def add_geometric_features(df):
    df = df.copy()
    # Euclidean distance
    df["distance"] = np.sqrt(
        (df["x_0"] - df["x_1"]) ** 2
        + (df["y_0"] - df["y_1"]) ** 2
        + (df["z_0"] - df["z_1"]) ** 2
    )

    # Distance squared and cubed
    df["distance_squared"] = df["distance"] ** 2
    df["distance_inv"] = 1 / (df["distance"] + 1e-6)

    # Coordinate differences
    df["dx"] = df["x_0"] - df["x_1"]
    df["dy"] = df["y_0"] - df["y_1"]
    df["dz"] = df["z_0"] - df["z_1"]

    return df


train = add_geometric_features(train)
test = add_geometric_features(test)


# Add molecular-level features
def add_molecular_features(df, dipole, potential, mulliken, magnetic):
    df = df.copy()

    # Merge dipole moments
    df = pd.merge(df, dipole, how="left", on="molecule_name")
    df = df.rename(columns={"X": "dipole_X", "Y": "dipole_Y", "Z": "dipole_Z"})

    # Merge potential energy
    df = pd.merge(df, potential, how="left", on="molecule_name")

    # Add mulliken charge features for both atoms
    mulliken_0 = mulliken.copy()
    mulliken_0 = mulliken_0.rename(
        columns={"atom_index": "atom_index_0", "mulliken_charge": "mulliken_charge_0"}
    )
    df = pd.merge(df, mulliken_0, how="left", on=["molecule_name", "atom_index_0"])

    mulliken_1 = mulliken.copy()
    mulliken_1 = mulliken_1.rename(
        columns={"atom_index": "atom_index_1", "mulliken_charge": "mulliken_charge_1"}
    )
    df = pd.merge(df, mulliken_1, how="left", on=["molecule_name", "atom_index_1"])

    # Add magnetic shielding tensor features
    magnetic_0 = magnetic.copy()
    magnetic_cols = ["XX", "XY", "XZ", "YX", "YY", "YZ", "ZX", "ZY", "ZZ"]
    for col in magnetic_cols:
        magnetic_0 = magnetic_0.rename(columns={col: f"{col}_0"})
    magnetic_0 = magnetic_0.rename(columns={"atom_index": "atom_index_0"})
    df = pd.merge(df, magnetic_0, how="left", on=["molecule_name", "atom_index_0"])

    magnetic_1 = magnetic.copy()
    for col in magnetic_cols:
        magnetic_1 = magnetic_1.rename(columns={col: f"{col}_1"})
    magnetic_1 = magnetic_1.rename(columns={"atom_index": "atom_index_1"})
    df = pd.merge(df, magnetic_1, how="left", on=["molecule_name", "atom_index_1"])

    # Calculate derived features
    df["mulliken_diff"] = df["mulliken_charge_0"] - df["mulliken_charge_1"]
    df["mulliken_sum"] = df["mulliken_charge_0"] + df["mulliken_charge_1"]
    df["mulliken_abs_diff"] = np.abs(df["mulliken_diff"])

    return df


print("Adding molecular features...")
train = add_molecular_features(train, dipole, potential, mulliken, magnetic)
test = add_molecular_features(test, dipole, potential, mulliken, magnetic)

# Add contributions for training data
if "scalar_coupling_constant" in train.columns:
    train = pd.merge(
        train,
        contributions,
        how="left",
        on=["molecule_name", "atom_index_0", "atom_index_1", "type"],
    )

# Add atomic properties
atomic_mass = {"H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998}
atomic_radius = {"H": 0.53, "C": 0.77, "N": 0.75, "O": 0.73, "F": 0.71}
electronegativity = {"H": 2.20, "C": 2.55, "N": 3.04, "O": 3.44, "F": 3.98}


def add_atomic_properties(df):
    df = df.copy()
    df["mass_0"] = df["atom_0"].map(atomic_mass)
    df["mass_1"] = df["atom_1"].map(atomic_mass)
    df["radius_0"] = df["atom_0"].map(atomic_radius)
    df["radius_1"] = df["atom_1"].map(atomic_radius)
    df["eneg_0"] = df["atom_0"].map(electronegativity)
    df["eneg_1"] = df["atom_1"].map(electronegativity)

    df["mass_sum"] = df["mass_0"] + df["mass_1"]
    df["mass_diff"] = df["mass_0"] - df["mass_1"]
    df["radius_sum"] = df["radius_0"] + df["radius_1"]
    df["eneg_diff"] = np.abs(df["eneg_0"] - df["eneg_1"])

    return df


train = add_atomic_properties(train)
test = add_atomic_properties(test)


# Add interaction type features
def add_type_features(df):
    df = df.copy()

    # Parse coupling type
    df["coupling_n"] = df["type"].str[0].astype(int)  # 1J, 2J, 3J
    df["coupling_atom1"] = df["type"].str[1]  # H, C, N
    df["coupling_atom2"] = df["type"].str[2]  # H, C, N

    # Check if atoms match expected pattern
    df["type_match_0"] = (df["atom_0"] == df["coupling_atom1"]).astype(int)
    df["type_match_1"] = (df["atom_1"] == df["coupling_atom2"]).astype(int)

    # Create atom pair type
    df["atom_pair"] = df["atom_0"] + "_" + df["atom_1"]

    return df


train = add_type_features(train)
test = add_type_features(test)

# Encode categorical features
categorical_cols = [
    "atom_0",
    "atom_1",
    "type",
    "atom_pair",
    "coupling_atom1",
    "coupling_atom2",
]

for col in categorical_cols:
    if col in train.columns:
        le = LabelEncoder()
        le.fit(pd.concat([train[col], test[col]], axis=0))
        train[col] = le.transform(train[col])
        test[col] = le.transform(test[col])

# Prepare features and target
features = [
    col
    for col in train.columns
    if col
    not in ["id", "molecule_name", "scalar_coupling_constant", "fc", "sd", "pso", "dso"]
    and train[col].dtype != object
]

X = train[features].copy()
y = train["scalar_coupling_constant"].copy()
X_test = test[features].copy()

# Handle NaN values
X = X.fillna(X.mean())
X_test = X_test.fillna(X.mean())

print(f"Training shape: {X.shape}, Test shape: {X_test.shape}")
print(f"Number of features: {len(features)}")

# Create out-of-fold predictions
n_folds = 5
kf = GroupKFold(n_splits=n_folds)
groups = train["molecule_name"]

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))
scores = []

# Train LightGBM model
print("\nTraining LightGBM model...")
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y, groups)):
    print(f"\nFold {fold + 1}/{n_folds}")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # Create LightGBM dataset
    train_set = lgb.Dataset(X_train, y_train)
    val_set = lgb.Dataset(X_val, y_val, reference=train_set)

    # Parameters optimized for this task
    params = {
        "objective": "regression_l1",  # MAE loss
        "metric": "mae",
        "boosting_type": "gbdt",
        "num_leaves": 127,
        "max_depth": -1,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 20,
        "min_child_weight": 0.001,
        "min_split_gain": 0.0,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "n_jobs": -1,
        "verbose": -1,
        "seed": 42 + fold,
    }

    # Train model with early stopping and logging callbacks
    callbacks = [
        lgb.early_stopping(stopping_rounds=100),
        lgb.log_evaluation(period=100),
    ]
    model = lgb.train(
        params,
        train_set,
        num_boost_round=2000,
        valid_sets=[train_set, val_set],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )

    # Make predictions
    oof_preds[val_idx] = model.predict(X_val)
    test_preds += model.predict(X_test) / n_folds

    # Calculate fold score
    fold_mae = np.mean(np.abs(y_val - oof_preds[val_idx]))
    scores.append(fold_mae)
    print(f"Fold {fold + 1} MAE: {fold_mae:.6f}")

# Calculate overall validation score
val_mae = np.mean(np.abs(y - oof_preds))
print(f"\nOverall Validation MAE: {val_mae:.6f}")
print(f"Fold MAEs: {[f'{s:.6f}' for s in scores]}")


# Calculate competition metric
def competition_score(y_true, y_pred, types):
    """Calculate the competition metric: log(MAE) averaged by coupling type"""
    results = []
    unique_types = np.unique(types)
    for type_name in unique_types:
        mask = types == type_name
        if mask.sum() > 0:
            mae = np.mean(np.abs(y_true[mask] - y_pred[mask]))
            mae = max(mae, 1e-9)  # Apply floor
            results.append(np.log(mae))
    return np.mean(results)


# Calculate competition metric using the type column from train
if "type" in train.columns:
    # Get the original type column (not encoded)
    type_col = train["type"].copy()
    # Ensure we use the same indices as our predictions
    score = competition_score(y, oof_preds, type_col)
    print(f"\nCompetition Score (log MAE): {score:.6f}")

# Create submission file
submission = pd.DataFrame({"id": test["id"], "scalar_coupling_constant": test_preds})

# Clip predictions to reasonable range based on training data
min_val = train["scalar_coupling_constant"].min()
max_val = train["scalar_coupling_constant"].max()
submission["scalar_coupling_constant"] = submission["scalar_coupling_constant"].clip(
    min_val, max_val
)

# Save submission
submission_path = SUBMISSION_PATH / "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"\nSubmission saved to: {submission_path}")
print(f"Submission shape: {submission.shape}")

# Validate submission format
print("\nFirst few predictions:")
print(submission.head())
print(
    f"\nPrediction stats: Mean={submission['scalar_coupling_constant'].mean():.4f}, "
    f"Std={submission['scalar_coupling_constant'].std():.4f}, "
    f"Min={submission['scalar_coupling_constant'].min():.4f}, "
    f"Max={submission['scalar_coupling_constant'].max():.4f}"
)

print("\nTraining completed successfully!")
