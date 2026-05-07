import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os


def add_domain_features(df):
    """Add domain-specific features to the dataframe."""
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
    wilderness_cols = [f'Wilderness_Area{i}' for i in range(1, 5)]
    df['Wilderness_Area_idx'] = (df[wilderness_cols].values * np.arange(1, 5)).sum(axis=1).astype(np.float32)
    df['Wilderness_Area_count'] = df[wilderness_cols].sum(axis=1).astype(np.float32)
    # Soil type features
    soil_cols = [f'Soil_Type{i}' for i in range(1, 41)]
    df['Soil_Type_idx'] = (df[soil_cols].values * np.arange(1, 41)).sum(axis=1).astype(np.float32)
    df['Soil_Type_count'] = df[soil_cols].sum(axis=1).astype(np.float32)
    # Aspect cyclic transformation
    df['Aspect_sin'] = np.sin(np.radians(df['Aspect'])).astype(np.float32)
    df['Aspect_cos'] = np.cos(np.radians(df['Aspect'])).astype(np.float32)
    return df


class MyDataLoader(BaseDataLoader):
    """Data loader for Forest Cover Type prediction dataset."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.label_encoder = None
        self.feature_cols = None
        self.test_ids = None
    
    def setup(self):
        """
        Load data, feature engineering, data augmentation, etc.
        Must set self.train_data and self.test_data
        """
        print("Loading data...")
        train_df = pd.read_csv('./input/train.csv')
        test_df = pd.read_csv('./input/test.csv')
        
        print("Feature engineering...")
        train_df = add_domain_features(train_df)
        test_df = add_domain_features(test_df)
        
        # Identify feature columns (all except Id and target)
        self.feature_cols = [col for col in train_df.columns 
                            if col not in ['Id', 'Cover_Type'] and col in test_df.columns]
        print(f"Number of features: {len(self.feature_cols)}")
        
        # Store test IDs for submission
        self.test_ids = test_df['Id'].values
        
        # Prepare features and target
        X = train_df[self.feature_cols].astype(np.float32).values
        y = train_df['Cover_Type'].values
        
        # Encode target to 0..6
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        X_test = test_df[self.feature_cols].astype(np.float32).values
        
        # Check for fixed validation set
        if os.path.exists('input/val.csv'):
            print("Using fixed validation set from input/val.csv")
            val_df = pd.read_csv('input/val.csv')
            val_df = add_domain_features(val_df)
            
            # Get validation features and labels
            X_valid = val_df[self.feature_cols].astype(np.float32).values
            y_valid = self.label_encoder.transform(val_df['Cover_Type'].values)
            
            # Remove validation samples from training data
            val_ids = set(val_df['Id'].values)
            train_mask = ~train_df['Id'].isin(val_ids)
            X_train = X[train_mask]
            y_train = y_encoded[train_mask]
        else:
            # Fallback to stratified split
            unique, counts = np.unique(y_encoded, return_counts=True)
            min_count = counts.min()
            if min_count >= 2:
                stratify = y_encoded
                print("Using stratified split.")
            else:
                stratify = None
                print(f"Warning: minimum class count is {min_count}, using random split (no stratification).")
            
            print("Splitting data...")
            X_train, X_valid, y_train, y_valid = train_test_split(
                X, y_encoded, test_size=0.1, stratify=stratify, shuffle=True, random_state=42
            )
        
        print(f"Train shape: {X_train.shape}, Valid shape: {X_valid.shape}")
        
        # Set train_data and test_data
        self.train_data = (X_train, y_train, X_valid, y_valid)
        self.test_data = (X_test, self.test_ids, self.label_encoder)
    
    def describe(self) -> str:
        """
        Return a description of your data processing approach
        """
        return """
        Data processing approach:
        1. Load train.csv and test.csv from input directory
        2. Apply domain-specific feature engineering:
           - Euclidean distance to hydrology
           - Elevation ± vertical distance combinations
           - Hillshade aggregates (mean, max, min, range)
           - Distance differences and sums between hydrology, roadways, and fire points
           - Euclidean combinations between distance pairs
           - Wilderness area features (index and count)
           - Soil type features (index and count)
           - Aspect cyclic transformation (sin/cos)
        3. Use fixed validation set from input/val.csv if available
        4. Otherwise, use stratified 90/10 train/validation split
        5. Label encode target variable (Cover_Type) to 0-6 range
        """

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier
import argparse
import os


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Forest Cover Type Prediction')
    
    # CatBoost parameters
    parser.add_argument('--cb_iterations', type=int, default=1200, 
                        help='CatBoost iterations (default: 1200)')
    parser.add_argument('--cb_learning_rate', type=float, default=0.07, 
                        help='CatBoost learning rate (default: 0.07)')
    parser.add_argument('--cb_depth', type=int, default=8, 
                        help='CatBoost depth (default: 8)')
    parser.add_argument('--cb_l2_leaf_reg', type=float, default=3.0, 
                        help='CatBoost L2 leaf regularization (default: 3.0)')
    parser.add_argument('--cb_subsample', type=float, default=0.8, 
                        help='CatBoost subsample (default: 0.8)')
    parser.add_argument('--random_seed', type=int, default=42, 
                        help='Random seed (default: 42)')
    
    # XGBoost parameters
    parser.add_argument('--xgb_n_estimators', type=int, default=700, 
                        help='XGBoost n_estimators (default: 700)')
    parser.add_argument('--xgb_learning_rate', type=float, default=0.08, 
                        help='XGBoost learning rate (default: 0.08)')
    parser.add_argument('--xgb_max_depth', type=int, default=8, 
                        help='XGBoost max depth (default: 8)')
    parser.add_argument('--xgb_subsample', type=float, default=0.8, 
                        help='XGBoost subsample (default: 0.8)')
    parser.add_argument('--xgb_colsample_bytree', type=float, default=0.8, 
                        help='XGBoost colsample_bytree (default: 0.8)')
    parser.add_argument('--xgb_reg_lambda', type=float, default=1.0, 
                        help='XGBoost L2 regularization (default: 1.0)')
    
    # Output paths
    parser.add_argument('--submission_path', type=str, default='./submission/submission.csv', 
                        help='Path to save submission (default: ./submission/submission.csv)')
    parser.add_argument('--working_path', type=str, default='./working/submission.csv', 
                        help='Path to save working copy (default: ./working/submission.csv)')
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Get data from DataLoader
    data_loader = MyDataLoader()
    train_data, test_data = data_loader.get_data()
    
    X_train, y_train, X_valid, y_valid = train_data
    X_test, test_ids, label_encoder = test_data
    
    model_name = None
    pred_valid = None
    pred_test = None
    
    # Try CatBoost GPU
    try:
        print("Attempting CatBoost GPU...")
        from catboost import CatBoostClassifier
        model = CatBoostClassifier(
            iterations=args.cb_iterations,
            learning_rate=args.cb_learning_rate,
            depth=args.cb_depth,
            l2_leaf_reg=args.cb_l2_leaf_reg,
            loss_function='MultiClass',
            classes_count=7,
            task_type='GPU',
            devices='0',
            bootstrap_type='Bernoulli',
            subsample=args.cb_subsample,
            random_seed=args.random_seed,
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
                n_estimators=args.xgb_n_estimators,
                learning_rate=args.xgb_learning_rate,
                max_depth=args.xgb_max_depth,
                subsample=args.xgb_subsample,
                colsample_bytree=args.xgb_colsample_bytree,
                reg_lambda=args.xgb_reg_lambda,
                tree_method='gpu_hist',
                predictor='gpu_predictor',
                random_state=args.random_seed,
                n_jobs=-1
            )
            model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
            model_name = "XGBoost"
            pred_valid = model.predict(X_valid)
            pred_test = model.predict(X_test)
        except Exception as e2:
            print(f"XGBoost GPU also failed: {e2}. Using fallback dummy model.")
            model = DummyClassifier(strategy='most_frequent')
            model.fit(X_train, y_train)
            model_name = "Dummy"
            pred_valid = model.predict(X_valid)
            pred_test = model.predict(X_test)
    
    # Compute validation accuracy
    acc = accuracy_score(y_valid, pred_valid)
    print(f"Validation Accuracy ({model_name}): {acc:.6f}")
    
    # Map predictions back to original class labels (1-7)
    pred_test_original = label_encoder.inverse_transform(pred_test)
    
    # Create submission
    submission = pd.DataFrame({'Id': test_ids, 'Cover_Type': pred_test_original})
    
    # Save to submission path
    submission_dir = os.path.dirname(args.submission_path)
    if submission_dir:
        os.makedirs(submission_dir, exist_ok=True)
    submission.to_csv(args.submission_path, index=False)
    print(f"Submission saved to {args.submission_path}")
    
    # Also save to working for compatibility
    working_dir = os.path.dirname(args.working_path)
    if working_dir:
        os.makedirs(working_dir, exist_ok=True)
    submission.to_csv(args.working_path, index=False)
    
    print("Done.")


if __name__ == "__main__":
    main()