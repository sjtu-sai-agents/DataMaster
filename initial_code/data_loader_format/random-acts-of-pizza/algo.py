import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='LightGBM training for RAOP prediction')
    
    # Model hyperparameters
    parser.add_argument('--num_leaves', type=int, default=31,
                        help='Number of leaves in LightGBM (default: 31)')
    parser.add_argument('--learning_rate', type=float, default=0.05,
                        help='Learning rate for boosting (default: 0.05)')
    parser.add_argument('--feature_fraction', type=float, default=0.9,
                        help='Feature fraction for each tree (default: 0.9)')
    parser.add_argument('--bagging_fraction', type=float, default=0.8,
                        help='Bagging fraction for each tree (default: 0.8)')
    parser.add_argument('--bagging_freq', type=int, default=5,
                        help='Bagging frequency (default: 5)')
    parser.add_argument('--num_boost_round', type=int, default=1000,
                        help='Number of boosting rounds (default: 1000)')
    parser.add_argument('--early_stopping_rounds', type=int, default=50,
                        help='Early stopping rounds (default: 50)')
    
    # Training parameters
    parser.add_argument('--n_splits', type=int, default=5,
                        help='Number of CV splits when no fixed validation set (default: 5)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--device', type=str, default='gpu', choices=['cpu', 'gpu'],
                        help='Device to use for training (default: gpu)')
    
    # Paths
    parser.add_argument('--output_path', type=str, default='./submission/submission.csv',
                        help='Output path for submission file (default: ./submission/submission.csv)')
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Ensure submission directory exists
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Load data using MyDataLoader
    print("Initializing data loader...")
    data_loader = MyDataLoader()
    train_data, test_data = data_loader.get_data()
    
    # Extract data
    X_train = train_data['X']
    y_train = train_data['y']
    X_test = test_data['X']
    test_ids = test_data['ids']
    has_val = train_data.get('has_val', False)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # LightGBM parameters
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': args.num_leaves,
        'learning_rate': args.learning_rate,
        'feature_fraction': args.feature_fraction,
        'bagging_fraction': args.bagging_fraction,
        'bagging_freq': args.bagging_freq,
        'verbose': -1,
        'num_threads': -1,
        'seed': args.seed,
        'device': args.device,
    }
    
    if has_val:
        # Use fixed validation set
        X_val = train_data['X_val']
        y_val = train_data['y_val']
        
        print("\nTraining with fixed validation set...")
        print(f"Train samples: {X_train.shape[0]}, Validation samples: {X_val.shape[0]}")
        
        lgb_train = lgb.Dataset(X_train, label=y_train)
        lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
        
        # Calculate scale_pos_weight for imbalanced data
        pos = y_train.sum()
        neg = len(y_train) - pos
        scale_pos_weight = neg / pos if pos > 0 else 1.0
        params_fold = params.copy()
        params_fold['scale_pos_weight'] = scale_pos_weight
        
        model = lgb.train(
            params_fold, 
            lgb_train, 
            num_boost_round=args.num_boost_round,
            valid_sets=[lgb_val],
            callbacks=[lgb.early_stopping(stopping_rounds=args.early_stopping_rounds, verbose=False)]
        )
        
        val_pred = model.predict(X_val)
        auc = roc_auc_score(y_val, val_pred)
        print(f"Validation AUC: {auc:.5f}")
        
        test_preds = model.predict(X_test)
        
    else:
        # Use cross-validation
        print("\nStarting cross-validation...")
        skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
        
        fold_aucs = []
        test_preds = np.zeros(len(X_test))
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            print(f"\nFold {fold+1}/{args.n_splits}")
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            
            lgb_train = lgb.Dataset(X_tr, label=y_tr)
            lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
            
            # Calculate scale_pos_weight for imbalanced data
            pos = y_tr.sum()
            neg = len(y_tr) - pos
            scale_pos_weight = neg / pos if pos > 0 else 1.0
            params_fold = params.copy()
            params_fold['scale_pos_weight'] = scale_pos_weight
            
            model = lgb.train(
                params_fold, 
                lgb_train, 
                num_boost_round=args.num_boost_round,
                valid_sets=[lgb_val],
                callbacks=[lgb.early_stopping(stopping_rounds=args.early_stopping_rounds, verbose=False)]
            )
            
            val_pred = model.predict(X_val)
            auc = roc_auc_score(y_val, val_pred)
            fold_aucs.append(auc)
            print(f"Fold {fold+1} AUC: {auc:.5f}")
            
            test_preds += model.predict(X_test)
        
        test_preds /= args.n_splits
        
        cv_mean = np.mean(fold_aucs)
        cv_std = np.std(fold_aucs)
        print(f"\nCV AUC: {cv_mean:.5f} ± {cv_std:.5f}")
    
    # Save submission
    submission = pd.DataFrame({
        'request_id': test_ids,
        'requester_received_pizza': test_preds
    })
    submission.to_csv(args.output_path, index=False)
    print(f"\nSubmission saved to {args.output_path}")


if __name__ == "__main__":
    main()