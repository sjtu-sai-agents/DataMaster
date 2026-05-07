import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import argparse
import os
import warnings

warnings.filterwarnings("ignore")
np.random.seed(42)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='NYC Taxi Fare Prediction with LightGBM')

    # Model hyperparameters
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='Learning rate for boosting (default: 0.1)')
    parser.add_argument('--num_leaves', type=int, default=127,
                        help='Maximum number of leaves in one tree (default: 127)')
    parser.add_argument('--max_depth', type=int, default=-1,
                        help='Maximum tree depth, -1 means no limit (default: -1)')
    parser.add_argument('--min_child_samples', type=int, default=20,
                        help='Minimum number of data points in one leaf (default: 20)')
    parser.add_argument('--subsample', type=float, default=0.8,
                        help='Subsample ratio of training data (default: 0.8)')
    parser.add_argument('--colsample_bytree', type=float, default=0.8,
                        help='Subsample ratio of columns when constructing each tree (default: 0.8)')
    parser.add_argument('--reg_alpha', type=float, default=0.1,
                        help='L1 regularization term (default: 0.1)')
    parser.add_argument('--reg_lambda', type=float, default=0.1,
                        help='L2 regularization term (default: 0.1)')
    parser.add_argument('--num_boost_round', type=int, default=1000,
                        help='Number of boosting iterations (default: 1000)')
    parser.add_argument('--early_stopping_rounds', type=int, default=50,
                        help='Early stopping rounds (default: 50)')

    # Path parameters
    parser.add_argument('--train_path', type=str, default='./input/labels.csv',
                        help='Path to training data (default: ./input/labels.csv)')
    parser.add_argument('--test_path', type=str, default='./input/test.csv',
                        help='Path to test data (default: ./input/test.csv)')
    parser.add_argument('--submission_path', type=str, default='./submission/submission.csv',
                        help='Path to save submission file (default: ./submission/submission.csv)')

    # Other parameters
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--device', type=str, default='gpu',
                        help='Device to use: cpu or gpu (default: gpu)')
    parser.add_argument('--max_rows', type=int, default=5000000,
                        help='Maximum rows to load from training data (default: 5000000)')

    # Prediction clipping
    parser.add_argument('--pred_min', type=float, default=2.5,
                        help='Minimum prediction value (default: 2.5)')
    parser.add_argument('--pred_max', type=float, default=200.0,
                        help='Maximum prediction value (default: 200.0)')

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    # Initialize data loader with configuration
    data_loader = MyDataLoader(max_rows=args.max_rows)

    # Get processed data
    train_data, test_data = data_loader.get_data()

    X_train = train_data['X_train']
    y_train = train_data['y_train']
    X_val = train_data['X_val']
    y_val = train_data['y_val']
    cat_features = train_data['cat_features']

    X_test = test_data['X_test']
    test_original = test_data['test_original']

    # LightGBM parameters
    params = {
        "boosting_type": "gbdt",
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "max_depth": args.max_depth,
        "min_child_samples": args.min_child_samples,
        "subsample": args.subsample,
        "subsample_freq": 1,
        "colsample_bytree": args.colsample_bytree,
        "reg_alpha": args.reg_alpha,
        "reg_lambda": args.reg_lambda,
        "n_jobs": -1,
        "device": args.device,
        "gpu_platform_id": 0,
        "gpu_device_id": 0,
        "verbose": -1,
        "seed": args.seed,
    }

    # Create LightGBM datasets
    train_lgb = lgb.Dataset(
        X_train, label=y_train, categorical_feature=cat_features, free_raw_data=False
    )
    val_lgb = lgb.Dataset(
        X_val, label=y_val, categorical_feature=cat_features, free_raw_data=False
    )

    # Train model
    print("Training LightGBM model...")
    print(f"Parameters: {params}")
    model = lgb.train(
        params,
        train_lgb,
        valid_sets=[train_lgb, val_lgb],
        valid_names=["train", "val"],
        num_boost_round=args.num_boost_round,
        callbacks=[
            lgb.early_stopping(stopping_rounds=args.early_stopping_rounds, verbose=True),
            lgb.log_evaluation(period=100),
        ],
    )

    # Evaluate on validation set
    y_pred = model.predict(X_val, num_iteration=model.best_iteration)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    print(f"Validation RMSE: {rmse:.4f}")

    # Make predictions on test set
    print("Making test predictions...")
    test_pred = model.predict(X_test, num_iteration=model.best_iteration)

    # Post-processing: clip to realistic range
    test_pred = np.clip(test_pred, args.pred_min, args.pred_max)

    # Create submission file
    submission = pd.DataFrame({"key": test_original["key"], "fare_amount": test_pred})

    os.makedirs(os.path.dirname(args.submission_path), exist_ok=True)
    submission.to_csv(args.submission_path, index=False)
    print(f"Submission saved to {args.submission_path}")

    # Print sample predictions
    print("\nSample predictions:")
    print(submission.head(10))

    return rmse


if __name__ == "__main__":
    final_rmse = main()
    print(f"\nFinal Validation RMSE: {final_rmse:.4f}")