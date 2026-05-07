import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import warnings

warnings.filterwarnings("ignore")


def advanced_f27_engineering(df):
    """Enhanced f_27 feature engineering"""
    df = df.copy()
    df["f_27_str"] = df["f_27"].astype(str)

    # Positional encoding
    for i in range(10):
        df[f"f_27_pos_{i}"] = df["f_27_str"].str[i].apply(lambda x: ord(x) - ord("A"))

    # Character frequency features
    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for char in chars:
        df[f"f_27_{char}_count"] = df["f_27_str"].str.count(char)

    # String statistics
    df["f_27_len"] = df["f_27_str"].str.len()
    df["f_27_unique"] = df["f_27_str"].apply(lambda x: len(set(str(x))))

    # Positional statistics
    pos_cols = [f"f_27_pos_{i}" for i in range(10)]
    pos_df = df[pos_cols]
    df["f_27_variance"] = pos_df.var(axis=1)
    df["f_27_mean"] = pos_df.mean(axis=1)
    df["f_27_std"] = pos_df.std(axis=1)
    df["f_27_min"] = pos_df.min(axis=1)
    df["f_27_max"] = pos_df.max(axis=1)
    df["f_27_range"] = df["f_27_max"] - df["f_27_min"]

    # Pattern features
    df["f_27_first_last_eq"] = (df["f_27_pos_0"] == df["f_27_pos_9"]).astype(int)
    df["f_27_is_palindrome"] = (
        (df[pos_cols] == df[pos_cols[::-1]].values).all(axis=1).astype(int)
    )

    # Drop original columns
    df.drop(["f_27", "f_27_str"], axis=1, inplace=True)
    return df


class MyDataLoader(BaseDataLoader):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def setup(self):
        """Load data, feature engineering, data augmentation, etc."""
        print("Loading data...")
        train = pd.read_csv("./input/train.csv")
        test = pd.read_csv("./input/test.csv")

        # Check for validation set - MUST use fixed val.csv if exists
        has_val = os.path.exists("./input/val.csv")
        X_val = None
        y_val = None
        
        if has_val:
            print("Found validation set, loading val.csv...")
            val = pd.read_csv("./input/val.csv")
            val_ids = set(val['id'].values)
            # Remove validation samples from train
            train = train[~train['id'].isin(val_ids)]
            X_val = val.drop(["id", "target"], axis=1).copy()
            y_val = val["target"].copy().values

        # Separate features and target
        X = train.drop(["id", "target"], axis=1).copy()
        y = train["target"].copy().values
        X_test = test.drop("id", axis=1).copy()
        test_ids = test["id"].copy()

        # Enhanced f_27 feature engineering
        print("Processing f_27 feature with advanced engineering...")
        X = advanced_f27_engineering(X)
        X_test = advanced_f27_engineering(X_test)
        if has_val:
            X_val = advanced_f27_engineering(X_val)

        # Align columns between train and test
        print("Aligning columns between train and test...")
        common_cols = X.columns.intersection(X_test.columns)
        X = X[common_cols]
        X_test = X_test[common_cols]
        if has_val:
            # Ensure val has same columns
            for col in common_cols:
                if col not in X_val.columns:
                    X_val[col] = 0
            X_val = X_val[common_cols]

        # Feature interaction engineering for continuous features
        print("Creating advanced feature interactions...")
        continuous_features = [
            f"f_{i:02d}" for i in list(range(7)) + list(range(19, 27)) + [28]
        ]
        continuous_features = [col for col in continuous_features if col in X.columns]

        # Create polynomial features for top correlated features
        if len(continuous_features) > 1:
            corr_matrix = X[continuous_features].corr().abs()
            top_pairs = []
            for i in range(len(continuous_features)):
                for j in range(i + 1, len(continuous_features)):
                    feat1, feat2 = continuous_features[i], continuous_features[j]
                    if abs(corr_matrix.loc[feat1, feat2]) > 0.1:
                        top_pairs.append((feat1, feat2))

            for feat1, feat2 in top_pairs[:50]:
                X[f"{feat1}_mul_{feat2}"] = X[feat1] * X[feat2]
                X[f"{feat1}_div_{feat2}"] = X[feat1] / (X[feat2] + 1e-8)
                X_test[f"{feat1}_mul_{feat2}"] = X_test[feat1] * X_test[feat2]
                X_test[f"{feat1}_div_{feat2}"] = X_test[feat1] / (X_test[feat2] + 1e-8)
                if has_val:
                    X_val[f"{feat1}_mul_{feat2}"] = X_val[feat1] * X_val[feat2]
                    X_val[f"{feat1}_div_{feat2}"] = X_val[feat1] / (X_val[feat2] + 1e-8)

        # Ensure column alignment again after creating new features
        common_cols = X.columns.intersection(X_test.columns)
        X = X[common_cols]
        X_test = X_test[common_cols]
        if has_val:
            for col in common_cols:
                if col not in X_val.columns:
                    X_val[col] = 0
            X_val = X_val[common_cols]

        # Identify categorical and continuous columns
        cat_cols = [f"f_{i:02d}" for i in range(7, 19)] + ["f_29", "f_30"]
        cat_cols = [col for col in cat_cols if col in X.columns]
        cont_cols = [col for col in X.columns if col not in cat_cols]

        # Label encode categorical features
        print("Label encoding categorical features...")
        le_dict = {}
        for col in cat_cols:
            le = LabelEncoder()
            combined = pd.concat([X[col], X_test[col]], axis=0)
            if has_val:
                combined = pd.concat([combined, X_val[col]], axis=0)
            le.fit(combined)
            X[col] = le.transform(X[col])
            X_test[col] = le.transform(X_test[col])
            if has_val:
                X_val[col] = le.transform(X_val[col])
            le_dict[col] = le

        # Scale continuous features using training data only
        print("Scaling continuous features...")
        scaler = StandardScaler()
        X[cont_cols] = scaler.fit_transform(X[cont_cols])
        X_test[cont_cols] = scaler.transform(X_test[cont_cols])
        if has_val:
            X_val[cont_cols] = scaler.transform(X_val[cont_cols])

        # Get categorical dimensions from LabelEncoder classes
        cat_dims = [len(le_dict[col].classes_) for col in cat_cols]
        n_cont = len(cont_cols)

        print(f"Categorical dimensions: {cat_dims}")
        print(f"Number of continuous features: {n_cont}")

        # Store processed data
        self.train_data = {
            'X': X,
            'y': y,
            'X_val': X_val,
            'y_val': y_val,
            'has_val': has_val,
            'cat_cols': cat_cols,
            'cont_cols': cont_cols,
            'cat_dims': cat_dims,
            'n_cont': n_cont,
        }

        self.test_data = {
            'X_test': X_test,
            'test_ids': test_ids,
        }

    def describe(self) -> str:
        """Return a description of your data processing approach"""
        return """
        Data Processing Pipeline:
        1. Load train.csv and test.csv (and val.csv if exists)
        2. Advanced f_27 feature engineering:
           - Positional encoding (10 positions)
           - Character frequency features (A-Z)
           - String statistics (length, unique chars)
           - Positional statistics (variance, mean, std, min, max, range)
           - Pattern features (first/last equality, palindrome check)
        3. Feature interaction engineering:
           - Polynomial features for correlated continuous features
           - Multiplication and division interactions
        4. Label encoding for categorical features
        5. Standard scaling for continuous features
        6. Fixed validation set from val.csv (if exists)
        """