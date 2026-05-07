import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


class MyDataLoader(BaseDataLoader):
    """
    Data loader for Spooky Author Identification task.
    Handles data loading, label encoding, and train/val/holdout splitting.
    Uses input/val.csv for validation if available.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.label_encoder = None
        self.num_labels = None
    
    def setup(self):
        """
        Load data, encode labels, and split into train/val/holdout sets.
        Uses input/val.csv for validation if it exists.
        """
        BASE_SEED = 42
        
        # Load data
        train_df = pd.read_csv("input/train.csv")
        test_df = pd.read_csv("input/test.csv")
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        train_df["author_code"] = self.label_encoder.fit_transform(train_df["author"])
        self.num_labels = len(self.label_encoder.classes_)
        
        # Check for validation file
        if os.path.exists('input/val.csv'):
            val_df = pd.read_csv('input/val.csv')
            
            # Check if val.csv has the full data with author column
            if 'author' in val_df.columns:
                # val.csv has the full data
                val_df["author_code"] = self.label_encoder.transform(val_df["author"])
                X_val = val_df["text"].values
                y_val = val_df["author_code"].values
                
                # Remove validation samples from training data
                if 'id' in val_df.columns and 'id' in train_df.columns:
                    val_ids = set(val_df['id'].values)
                    train_data = train_df[~train_df['id'].isin(val_ids)]
                else:
                    val_texts = set(val_df['text'].values)
                    train_data = train_df[~train_df['text'].isin(val_texts)]
            else:
                # val.csv only has identifier column
                if 'id' in val_df.columns and 'id' in train_df.columns:
                    val_ids = set(val_df['id'].values)
                    val_data = train_df[train_df['id'].isin(val_ids)]
                    train_data = train_df[~train_df['id'].isin(val_ids)]
                elif 'text' in val_df.columns:
                    val_texts = set(val_df['text'].values)
                    val_data = train_df[train_df['text'].isin(val_texts)]
                    train_data = train_df[~train_df['text'].isin(val_texts)]
                else:
                    raise ValueError("val.csv must have 'id' or 'text' column")
                
                X_val = val_data["text"].values
                y_val = val_data["author_code"].values
            
            # Create holdout from remaining training data
            try:
                train_sub, holdout_data = train_test_split(
                    train_data,
                    test_size=0.1,
                    stratify=train_data["author_code"],
                    random_state=BASE_SEED
                )
            except ValueError:
                # Fall back to non-stratified splitting if too few samples
                train_sub, holdout_data = train_test_split(
                    train_data,
                    test_size=0.1,
                    random_state=BASE_SEED
                )
        else:
            # Original splitting logic: holdout (10%) + train_data (90%)
            train_data, holdout_data = train_test_split(
                train_df,
                test_size=0.1,
                stratify=train_df["author_code"],
                random_state=BASE_SEED
            )
            
            # Split train_data into train_sub (80%) and val_sub (20%)
            train_sub, val_sub = train_test_split(
                train_data,
                test_size=0.2,
                stratify=train_data["author_code"],
                random_state=BASE_SEED
            )
            
            X_val = val_sub["text"].values
            y_val = val_sub["author_code"].values
        
        # Extract texts and labels
        X_train = train_sub["text"].values
        y_train = train_sub["author_code"].values
        X_holdout = holdout_data["text"].values
        y_holdout = holdout_data["author_code"].values
        X_test = test_df["text"].values
        test_ids = test_df["id"].values
        
        # Store processed data
        self.train_data = {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_holdout': X_holdout,
            'y_holdout': y_holdout,
            'num_labels': self.num_labels
        }
        
        self.test_data = {
            'X_test': X_test,
            'test_ids': test_ids
        }
    
    def describe(self) -> str:
        """
        Return a description of the data processing approach.
        """
        return ("Data loader for Spooky Author Identification task. "
                "Loads train/test CSV files, encodes author labels using LabelEncoder, "
                "and splits data into train/validation/holdout sets. "
                "Uses input/val.csv for validation if available (fixed validation set), "
                "otherwise creates stratified splits (72% train, 18% val, 10% holdout). "
                "Returns raw text data ready for tokenization.")