import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import os

def add_domain_features(df):
    df = df.copy()
    # Euclidean distance to hydrology
    df['Dist_To_Hydrology'] = np.sqrt(df['Horizontal_Distance_To_Hydrology']**2 + df['Vertical_Distance_To_Hydrology']**2).astype(np.float32)
    # Elevation ± vertical distance
    df['Elevation_Hydrology_Sum'] = (df['Elevation'] + df['Vertical_Distance_To_Hydrology']).astype(np.float32)
    df['Elevation_Hydrology_Diff'] = (df['Elevation'] - df['Vertical_Distance_To_Hydrology']).astype(np.float32)
    # Hillshade aggregates
    hillshade_cols = ['Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm']
    df['Hillshade_Mean'] = df[hillshade_cols].mean(axis=1).astype(np.float32)
    df['Hillshade_Max'] = df[hillshade_cols].max(axis=1).astype(np.float32)
    df['Hillshade_Min'] = df[hillshade_cols].min(axis=1).astype(np.float32)
    df['Hillshade_Range'] = (df['Hillshade_Max'] - df['Hillshade_Min']).astype(np.float32)
    # Distance differences and sums
    df['Hydro_Road_Diff'] = (df['Horizontal_Distance_To_Hydrology'] - df['Horizontal_Distance_To_Roadways']).abs().astype(np.float32)
    df['Hydro_Fire_Diff'] = (df['Horizontal_Distance_To_Hydrology'] - df['Horizontal_Distance_To_Fire_Points']).abs().astype(np.float32)
    df['Road_Fire_Diff'] = (df['Horizontal_Distance_To_Roadways'] - df['Horizontal_Distance_To_Fire_Points']).abs().astype(np.float32)
    df['Hydro_Road_Sum'] = (df['Horizontal_Distance_To_Hydrology'] + df['Horizontal_Distance_To_Roadways']).astype(np.float32)
    df['Hydro_Fire_Sum'] = (df['Horizontal_Distance_To_Hydrology'] + df['Horizontal_Distance_To_Fire_Points']).astype(np.float32)
    df['Road_Fire_Sum'] = (df['Horizontal_Distance_To_Roadways'] + df['Horizontal_Distance_To_Fire_Points']).astype(np.float32)
    # Euclidean combinations between pairs
    df['Euc_Hydro_Road'] = np.sqrt(df['Horizontal_Distance_To_Hydrology']**2 + df['Horizontal_Distance_To_Roadways']**2).astype(np.float32)
    df['Euc_Hydro_Fire'] = np.sqrt(df['Horizontal_Distance_To_Hydrology']**2 + df['Horizontal_Distance_To_Fire_Points']**2).astype(np.float32)
    df['Euc_Road_Fire'] = np.sqrt(df['Horizontal_Distance_To_Roadways']**2 + df['Horizontal_Distance_To_Fire_Points']**2).astype(np.float32)
    # Wilderness area features
    wilderness_cols = [f'Wilderness_Area{i}' for i in range(1,5)]
    df['Wilderness_Area_idx'] = (df[wilderness_cols].values * np.arange(1,5)).sum(axis=1).astype(np.float32)
    df['Wilderness_Area_count'] = df[wilderness_cols].sum(axis=1).astype(np.float32)
    # Soil type features
    soil_cols = [f'Soil_Type{i}' for i in range(1,41)]
    df['Soil_Type_idx'] = (df[soil_cols].values * np.arange(1,41)).sum(axis=1).astype(np.float32)
    df['Soil_Type_count'] = df[soil_cols].sum(axis=1).astype(np.float32)
    # Aspect cyclic transformation
    df['Aspect_sin'] = np.sin(np.radians(df['Aspect'])).astype(np.float32)
    df['Aspect_cos'] = np.cos(np.radians(df['Aspect'])).astype(np.float32)
    return df

print("Loading data...")
train_df = pd.read_csv('./input/train.csv')
test_df = pd.read_csv('./input/test.csv')

print("Feature engineering...")
train_df = add_domain_features(train_df)
test_df = add_domain_features(test_df)

# Identify feature columns (all except Id and target)
feature_cols = [col for col in train_df.columns if col not in ['Id', 'Cover_Type'] and col in test_df.columns]
print(f"Number of features: {len(feature_cols)}")

# Prepare features and target
X = train_df[feature_cols].astype(np.float32)
y = train_df['Cover_Type'].values

# Encode target to 0..6
le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_test = test_df[feature_cols].astype(np.float32)

# Determine if stratified split is possible
unique, counts = np.unique(y_encoded, return_counts=True)
min_count = counts.min()
if min_count >= 2:
    stratify = y_encoded
    print("Using stratified split.")
else:
    stratify = None
    print(f"Warning: minimum class count is {min_count}, using random split (no stratification).")

# Split (90/10)
print("Splitting data...")
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y_encoded, test_size=0.1, stratify=stratify, shuffle=True, random_state=42
)

print(f"Train shape: {X_train.shape}, Valid shape: {X_valid.shape}")

model_name = None
pred_valid = None
pred_test = None

# Try CatBoost GPU
try:
    print("Attempting CatBoost GPU...")
    from catboost import CatBoostClassifier
    model = CatBoostClassifier(
        iterations=1200,
        learning_rate=0.07,
        depth=8,
        l2_leaf_reg=3.0,
        loss_function='MultiClass',
        classes_count=7,
        task_type='GPU',
        devices='0',
        bootstrap_type='Bernoulli',
        subsample=0.8,
        random_seed=42,
        verbose=False
    )
    model.fit(X_train, y_train, eval_set=(X_valid, y_valid), use_best_model=False)
    model_name = "CatBoost"
    pred_valid = model.predict(X_valid, prediction_type='Class').ravel()
    pred_test = model.predict(X_test, prediction_type='Class').ravel()
except Exception as e:
    print(f"CatBoost GPU failed: {e}. Falling back to XGBoost GPU.")
    try:
        import xgboost as xgb
        model = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=7,
            n_estimators=700,
            learning_rate=0.08,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            tree_method='gpu_hist',
            predictor='gpu_predictor',
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
        model_name = "XGBoost"
        pred_valid = model.predict(X_valid)
        pred_test = model.predict(X_test)
    except Exception as e2:
        print(f"XGBoost GPU also failed: {e2}. Using fallback dummy model.")
        from sklearn.dummy import DummyClassifier
        model = DummyClassifier(strategy='most_frequent')
        model.fit(X_train, y_train)
        model_name = "Dummy"
        pred_valid = model.predict(X_valid)
        pred_test = model.predict(X_test)

# Compute validation accuracy
acc = accuracy_score(y_valid, pred_valid)
print(f"Validation Accuracy ({model_name}): {acc:.6f}")

# Map predictions back to original class labels (1-7)
pred_test_original = le.inverse_transform(pred_test)

# Create submission
submission = pd.DataFrame({'Id': test_df['Id'], 'Cover_Type': pred_test_original})
submission_path = './submission/submission.csv'
os.makedirs('./submission', exist_ok=True)
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")

# Also save to working for compatibility (optional)
os.makedirs('./working', exist_ok=True)
submission.to_csv('./working/submission.csv', index=False)

print("Done.")