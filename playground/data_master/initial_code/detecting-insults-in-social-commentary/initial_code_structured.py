import os
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


# ----------------------------------------------------------------------
# BaseDataLoader definition (DO NOT MODIFY)
# ----------------------------------------------------------------------
class BaseDataLoader(ABC):
    """
    Abstract base class for data loaders.

    This class defines the interface that all data loaders must implement.
    Subclasses only need to implement:
    - setup(): Load and process data
    - describe(): Provide a description

    Data splitting is NOT handled here - it's the responsibility of the training code.

    Attributes:
        config: Additional keyword arguments for customization
        train_data: Processed training data (set by setup())
        test_data: Processed test data (set by setup())
    """

    def __init__(self, **kwargs):
        """
        Initialize the BaseDataLoader.

        Args:
            **kwargs: Additional configuration (accessible via self.config)
        """
        self.config = kwargs
        self.train_data = None
        self.test_data = None
        self._is_setup = False

    # ========================================================================
    # Abstract methods - must be implemented by subclasses
    # ========================================================================

    @abstractmethod
    def setup(self):
        """
        Setup the data loader.

        Subclasses must implement this method to:
        - Load data from disk (CSV, images, etc.)
        - Perform feature engineering
        - Define data augmentation strategies
        - Set self.train_data and self.test_data

        Note: Do NOT perform train/val split here. Splitting is handled by the training code.
        """
        raise NotImplementedError(
            "Methods should implement `setup` methods for specific data loader class!"
        )

    @abstractmethod
    def describe(self) -> str:
        """
        Return a description of this data loader.

        Should describe:
        - What task this loader is for
        - What data processing/augmentation tricks is applied
        - What new data source and datsets are added.
        - Any notable features of the implementation

        Returns:
            A string description of the data loader
        """
        raise NotImplementedError(
            "Methods should implement `describe` methods for specific data loader class!"
        )

    def get_data(self):
        """
        Return the processed training and test data.

        This is the main interface - returns the data after all
        preprocessing, feature engineering, and augmentation.

        Automatically calls setup() if not already called.
        """
        if not self._is_setup:
            print(
                "Warning: Using Base Methods get_data: load self.train_data & self.test_data"
            )
            self.setup()
            self._is_setup = True

        return self.train_data, self.test_data

    def __str__(self) -> str:
        return self.describe()


SEED = 42
TRAIN_PATH = "./input/train.csv"
TEST_PATH = "./input/test.csv"
SUBMISSION_PATH = "./submission/submission.csv"


class MyDataLoader(BaseDataLoader):
    """Data loader for the Detecting Insults in Social Commentary competition."""

    def setup(self):
        np.random.seed(SEED)

        train_df = pd.read_csv(TRAIN_PATH)
        test_df = pd.read_csv(TEST_PATH)

        if "Insult" in train_df.columns:
            y = train_df["Insult"].to_numpy()
            X_text = train_df["Comment"].fillna("").astype(str).to_numpy()
        else:
            y = train_df.iloc[:, 0].to_numpy()
            X_text = train_df.iloc[:, 2].fillna("").astype(str).to_numpy()

        X_train_text, X_val_text, y_train, y_val = train_test_split(
            X_text,
            y,
            test_size=0.2,
            random_state=SEED,
            stratify=y,
        )

        vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
        )
        X_train = vectorizer.fit_transform(X_train_text)
        X_val = vectorizer.transform(X_val_text)

        if "Comment" in test_df.columns:
            test_text = test_df["Comment"].fillna("").astype(str).to_numpy()
            test_date = test_df["Date"].fillna("").astype(str).to_numpy()
            test_comment = test_df["Comment"].fillna("").astype(str).to_numpy()
        else:
            test_text = test_df.iloc[:, 2].fillna("").astype(str).to_numpy()
            test_date = test_df.iloc[:, 1].fillna("").astype(str).to_numpy()
            test_comment = test_df.iloc[:, 2].fillna("").astype(str).to_numpy()

        X_test = vectorizer.transform(test_text)

        self.train_data = {
            "X_train": X_train,
            "X_val": X_val,
            "y_train": y_train,
            "y_val": y_val,
        }
        self.test_data = {
            "X_test": X_test,
            "test_date": test_date,
            "test_comment": test_comment,
        }

    def describe(self) -> str:
        return (
            "Loads insult-comment text data, performs the original TF-IDF feature "
            "extraction, and returns train/validation/test matrices for the baseline model."
        )

    def get_data(self):
        if not self._is_setup:
            self.setup()
            self._is_setup = True
        return self.train_data, self.test_data


def main():
    loader = MyDataLoader()
    train_data, test_data = loader.get_data()

    X_train = train_data["X_train"]
    X_val = train_data["X_val"]
    y_train = train_data["y_train"]
    y_val = train_data["y_val"]

    X_test = test_data["X_test"]
    test_date = test_data["test_date"]
    test_comment = test_data["test_comment"]

    model = LogisticRegression(
        C=1.0,
        solver="liblinear",
        random_state=SEED,
        max_iter=1000,
    )
    model.fit(X_train, y_train)

    val_preds = model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_preds)
    print(f"Validation AUC: {val_auc:.4f}")

    test_preds = model.predict_proba(X_test)[:, 1]

    submission_df = pd.DataFrame(
        {
            "Insult": test_preds,
            "Date": test_date,
            "Comment": test_comment,
        }
    )

    os.makedirs("./submission", exist_ok=True)
    submission_df.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH} with shape {submission_df.shape}")


if __name__ == "__main__":
    main()
