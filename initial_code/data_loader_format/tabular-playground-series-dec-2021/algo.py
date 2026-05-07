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