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