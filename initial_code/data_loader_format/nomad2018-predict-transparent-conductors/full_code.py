import os
import numpy as np
import pandas as pd


def add_features(df):
    """Feature engineering function for semiconductor properties prediction."""
    df = df.copy()
    # Oxygen percentage
    df['percent_atom_o'] = 1.0 - (df['percent_atom_al'] + df['percent_atom_ga'] + df['percent_atom_in'])
    # Lattice angles to radians
    df['lattice_angle_alpha_rad'] = np.radians(df['lattice_angle_alpha_degree'])
    df['lattice_angle_beta_rad'] = np.radians(df['lattice_angle_beta_degree'])
    df['lattice_angle_gamma_rad'] = np.radians(df['lattice_angle_gamma_degree'])
    # Cell volume (triclinic formula)
    a = df['lattice_vector_1_ang']
    b = df['lattice_vector_2_ang']
    c = df['lattice_vector_3_ang']
    cos_alpha = np.cos(df['lattice_angle_alpha_rad'])
    cos_beta = np.cos(df['lattice_angle_beta_rad'])
    cos_gamma = np.cos(df['lattice_angle_gamma_rad'])
    term = 1 + 2*cos_alpha*cos_beta*cos_gamma - cos_alpha**2 - cos_beta**2 - cos_gamma**2
    term = np.clip(term, 0, None)  # numerical safety
    df['cell_volume'] = a * b * c * np.sqrt(term)
    # Volume per atom
    df['volume_per_atom'] = df['cell_volume'] / df['number_of_total_atoms']
    # Squared fractions
    df['al_frac_sq'] = df['percent_atom_al'] ** 2
    df['ga_frac_sq'] = df['percent_atom_ga'] ** 2
    df['in_frac_sq'] = df['percent_atom_in'] ** 2
    df['o_frac_sq'] = df['percent_atom_o'] ** 2
    # Interaction with total atoms
    df['al_by_atoms'] = df['percent_atom_al'] * df['number_of_total_atoms']
    df['ga_by_atoms'] = df['percent_atom_ga'] * df['number_of_total_atoms']
    df['in_by_atoms'] = df['percent_atom_in'] * df['number_of_total_atoms']
    df['o_by_atoms'] = df['percent_atom_o'] * df['number_of_total_atoms']
    # Metal ratios (add epsilon to avoid division by zero)
    eps = 1e-6
    df['al_over_ga'] = df['percent_atom_al'] / (df['percent_atom_ga'] + eps)
    df['al_over_in'] = df['percent_atom_al'] / (df['percent_atom_in'] + eps)
    df['ga_over_in'] = df['percent_atom_ga'] / (df['percent_atom_in'] + eps)
    # Categorical features
    df['spacegroup'] = df['spacegroup'].astype('category')
    df['number_of_total_atoms'] = df['number_of_total_atoms'].astype('category')
    return df


class MyDataLoader(BaseDataLoader):
    """Data loader for semiconductor properties prediction."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_dir = kwargs.get('input_dir', './input')
        
    def setup(self):
        """
        Load data, perform feature engineering, and split train/validation.
        Uses val.csv for validation if available, otherwise falls back to random split.
        """
        # Load data
        train_df = pd.read_csv(os.path.join(self.input_dir, "train.csv"))
        test_df = pd.read_csv(os.path.join(self.input_dir, "test.csv"))
        
        # Apply feature engineering
        train_fe = add_features(train_df)
        test_fe = add_features(test_df)
        
        # Define feature columns (exclude id and targets)
        exclude_cols = ['id', 'formation_energy_ev_natom', 'bandgap_energy_ev']
        feature_cols = [col for col in train_fe.columns if col not in exclude_cols]
        
        # Prepare features and targets
        X = train_fe[feature_cols]
        y1 = np.log1p(train_fe['formation_energy_ev_natom'])
        y2 = np.log1p(train_fe['bandgap_energy_ev'])
        X_test = test_fe[feature_cols]
        test_ids = test_df['id']
        
        # Identify categorical columns for LightGBM
        cat_features = [col for col in feature_cols if train_fe[col].dtype.name == 'category']
        
        # Split train/val - check for val.csv first (CRITICAL: use fixed validation set)
        val_path = os.path.join(self.input_dir, "val.csv")
        if os.path.exists(val_path):
            val_df = pd.read_csv(val_path)
            val_ids = set(val_df['id'].values)
            
            # Check if val.csv has target columns
            if 'formation_energy_ev_natom' in val_df.columns:
                # val.csv has full data with targets
                val_fe = add_features(val_df)
                X_val = val_fe[feature_cols].reset_index(drop=True)
                y1_val = np.log1p(val_fe['formation_energy_ev_natom']).reset_index(drop=True)
                y2_val = np.log1p(val_fe['bandgap_energy_ev']).reset_index(drop=True)
            else:
                # val.csv has only ids, extract validation data from train
                val_mask = train_fe['id'].isin(val_ids)
                X_val = X.loc[val_mask].reset_index(drop=True)
                y1_val = y1.loc[val_mask].reset_index(drop=True)
                y2_val = y2.loc[val_mask].reset_index(drop=True)
            
            # Remove validation samples from training set
            train_mask = ~train_fe['id'].isin(val_ids)
            X_train = X.loc[train_mask].reset_index(drop=True)
            y1_train = y1.loc[train_mask].reset_index(drop=True)
            y2_train = y2.loc[train_mask].reset_index(drop=True)
        else:
            # Fallback to random split only if val.csv doesn't exist
            from sklearn.model_selection import train_test_split
            X_train, X_val, y1_train, y1_val = train_test_split(
                X, y1, test_size=0.2, random_state=42
            )
            _, _, y2_train, y2_val = train_test_split(
                X, y2, test_size=0.2, random_state=42
            )
        
        # Set train_data and test_data
        self.train_data = {
            'X_train': X_train,
            'X_val': X_val,
            'y1_train': y1_train,
            'y1_val': y1_val,
            'y2_train': y2_train,
            'y2_val': y2_val,
            'cat_features': cat_features,
            'feature_cols': feature_cols,
            'X_full': X,
            'y1_full': y1,
            'y2_full': y2
        }
        self.test_data = {
            'X_test': X_test,
            'test_ids': test_ids
        }
        
    def describe(self) -> str:
        """
        Return a description of the data processing approach.
        """
        return ("Data loader for semiconductor properties prediction. "
                "Features include: oxygen percentage, lattice angles (radians), "
                "cell volume (triclinic formula), volume per atom, squared fractions, "
                "metal ratios, and categorical encoding for spacegroup and number_of_total_atoms. "
                "Uses fixed validation set from input/val.csv if available.")

import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import argparse


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='LightGBM training for semiconductor properties prediction'
    )
    # Path arguments
    parser.add_argument('--input_dir', type=str, default='./input',
                        help='Input directory containing train.csv, test.csv, and val.csv')
    parser.add_argument('--output_dir', type=str, default='./submission',
                        help='Output directory for submission file')
    parser.add_argument('--working_dir', type=str, default='./working',
                        help='Working directory for intermediate files')
    
    # Model hyperparameters
    parser.add_argument('--learning_rate', type=float, default=0.03,
                        help='Learning rate for LightGBM')
    parser.add_argument('--subsample', type=float, default=0.9,
                        help='Subsample ratio of training data')
    parser.add_argument('--colsample_bytree', type=float, default=0.9,
                        help='Subsample ratio of columns when constructing each tree')
    parser.add_argument('--num_leaves', type=int, default=64,
                        help='Maximum number of leaves in one tree')
    parser.add_argument('--reg_lambda', type=float, default=0.0,
                        help='L2 regularization term')
    parser.add_argument('--reg_alpha', type=float, default=0.0,
                        help='L1 regularization term')
    
    # Training parameters
    parser.add_argument('--num_boost_round', type=int, default=5000,
                        help='Number of boosting iterations')
    parser.add_argument('--early_stopping_rounds', type=int, default=200,
                        help='Early stopping rounds')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed for reproducibility')
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.working_dir, exist_ok=True)
    
    # Get data from DataLoader
    data_loader = MyDataLoader(input_dir=args.input_dir)
    train_data, test_data = data_loader.get_data()
    
    # Extract training data
    X_train = train_data['X_train']
    X_val = train_data['X_val']
    y1_train = train_data['y1_train']
    y1_val = train_data['y1_val']
    y2_train = train_data['y2_train']
    y2_val = train_data['y2_val']
    cat_features = train_data['cat_features']
    X_full = train_data['X_full']
    y1_full = train_data['y1_full']
    y2_full = train_data['y2_full']
    
    # Extract test data
    X_test = test_data['X_test']
    test_ids = test_data['test_ids']
    
    # LightGBM parameters
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': args.learning_rate,
        'subsample': args.subsample,
        'colsample_bytree': args.colsample_bytree,
        'num_leaves': args.num_leaves,
        'reg_lambda': args.reg_lambda,
        'reg_alpha': args.reg_alpha,
        'random_state': args.random_state,
        'n_jobs': -1,
        'verbose': -1,
    }
    
    # Train formation energy model
    print("Training formation energy model...")
    train_set1 = lgb.Dataset(X_train, y1_train, categorical_feature=cat_features)
    val_set1 = lgb.Dataset(X_val, y1_val, reference=train_set1, categorical_feature=cat_features)
    model1 = lgb.train(
        params,
        train_set1,
        num_boost_round=args.num_boost_round,
        valid_sets=[val_set1],
        callbacks=[lgb.early_stopping(stopping_rounds=args.early_stopping_rounds, verbose=False)],
    )
    best_iter1 = model1.best_iteration
    val_pred1_log = model1.predict(X_val)
    val_rmsle1 = np.sqrt(mean_squared_error(y1_val, val_pred1_log))
    
    # Train bandgap energy model
    print("Training bandgap energy model...")
    train_set2 = lgb.Dataset(X_train, y2_train, categorical_feature=cat_features)
    val_set2 = lgb.Dataset(X_val, y2_val, reference=train_set2, categorical_feature=cat_features)
    model2 = lgb.train(
        params,
        train_set2,
        num_boost_round=args.num_boost_round,
        valid_sets=[val_set2],
        callbacks=[lgb.early_stopping(stopping_rounds=args.early_stopping_rounds, verbose=False)],
    )
    best_iter2 = model2.best_iteration
    val_pred2_log = model2.predict(X_val)
    val_rmsle2 = np.sqrt(mean_squared_error(y2_val, val_pred2_log))
    
    # Compute mean RMSLE
    val_rmsle_mean = (val_rmsle1 + val_rmsle2) / 2
    
    print(f"Validation RMSLE (formation): {val_rmsle1:.6f}")
    print(f"Validation RMSLE (bandgap):   {val_rmsle2:.6f}")
    print(f"Mean RMSLE:                   {val_rmsle_mean:.6f}")
    
    # Retrain on full training set with best iteration
    print("Retraining on full training set...")
    full_set1 = lgb.Dataset(X_full, y1_full, categorical_feature=cat_features)
    full_model1 = lgb.train(params, full_set1, num_boost_round=best_iter1)
    full_set2 = lgb.Dataset(X_full, y2_full, categorical_feature=cat_features)
    full_model2 = lgb.train(params, full_set2, num_boost_round=best_iter2)
    
    # Predict on test set
    test_pred1_log = full_model1.predict(X_test)
    test_pred2_log = full_model2.predict(X_test)
    # Clip to avoid overflow in expm1
    test_pred1_log = np.clip(test_pred1_log, -50, 50)
    test_pred2_log = np.clip(test_pred2_log, -50, 50)
    test_pred1 = np.expm1(test_pred1_log)
    test_pred2 = np.expm1(test_pred2_log)
    # Ensure non-negativity
    test_pred1 = np.clip(test_pred1, 0, None)
    test_pred2 = np.clip(test_pred2, 0, None)
    
    # Create submission dataframe
    submission = pd.DataFrame({
        'id': test_ids,
        'formation_energy_ev_natom': test_pred1,
        'bandgap_energy_ev': test_pred2,
    })
    submission.to_csv(os.path.join(args.output_dir, "submission.csv"), index=False)
    submission.to_csv(os.path.join(args.working_dir, "submission.csv"), index=False)
    
    print("Submission file saved successfully.")


if __name__ == "__main__":
    main()