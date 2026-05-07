import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from sklearn.model_selection import StratifiedKFold

# Set random seeds for reproducibility
RANDOM_SEED = 42
def set_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# -------------------------------
# Dataset
# -------------------------------
class ToxicDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_len=192):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        enc = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            add_special_tokens=True,
            return_tensors='pt'
        )
        input_ids = enc['input_ids'].squeeze(0)
        attention_mask = enc['attention_mask'].squeeze(0)

        if self.labels is not None:
            labels = torch.tensor(self.labels[idx], dtype=torch.float)
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels
            }
        else:
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }

# -------------------------------
# MyDataLoader
# -------------------------------
class MyDataLoader(BaseDataLoader):
    def __init__(self, model_name="microsoft/deberta-v3-base", max_len=192, 
                 n_folds=5, random_seed=42, **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.max_len = max_len
        self.n_folds = n_folds
        self.random_seed = random_seed
        self.label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
        
    def setup(self):
        """
        Load data, feature engineering, data augmentation, etc.
        Must set self.train_data and self.test_data
        """
        set_seed(self.random_seed)
        
        # Load data
        train_df = pd.read_csv('./input/train.csv')
        test_df = pd.read_csv('./input/test.csv')
        
        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Handle validation set - use fixed val.csv if exists
        has_val_csv = False
        if os.path.exists('./input/val.csv'):
            has_val_csv = True
            val_df = pd.read_csv('./input/val.csv')
            val_ids = set(val_df['id'].values)
            # Remove val samples from train
            train_df = train_df[~train_df['id'].isin(val_ids)]
            # Use StratifiedKFold for remaining training data (folds 1 to n_folds-1)
            n_splits = self.n_folds - 1
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_seed)
            train_df['fold'] = -1
            for fold, (_, val_idx) in enumerate(skf.split(train_df, train_df[self.label_cols].sum(axis=1)), start=1):
                train_df.loc[train_df.iloc[val_idx].index, 'fold'] = fold
            # Assign fold 0 to val samples
            val_df['fold'] = 0
            val_df['is_external_val'] = True
            train_df['is_external_val'] = False
            # Combine
            train_df = pd.concat([train_df, val_df], ignore_index=True)
        else:
            # Use StratifiedKFold
            skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_seed)
            train_df['fold'] = -1
            for fold, (_, val_idx) in enumerate(skf.split(train_df, train_df[self.label_cols].sum(axis=1))):
                train_df.loc[train_df.iloc[val_idx].index, 'fold'] = fold
            train_df['is_external_val'] = False
        
        self.train_data = {
            'train_df': train_df,
            'tokenizer': tokenizer,
            'label_cols': self.label_cols,
            'max_len': self.max_len,
            'n_folds': self.n_folds,
            'has_val_csv': has_val_csv
        }
        
        self.test_data = {
            'test_df': test_df,
            'tokenizer': tokenizer,
            'label_cols': self.label_cols,
            'max_len': self.max_len
        }
        
    def describe(self) -> str:
        """
        Return a description of your data processing approach
        """
        return f"""
        Toxic Comment Classification DataLoader
        
        - Model: {self.model_name}
        - Max sequence length: {self.max_len}
        - Number of folds: {self.n_folds}
        - Random seed: {self.random_seed}
        - Label columns: {self.label_cols}
        
        Data processing:
        - Loads train.csv and test.csv
        - Uses fixed validation set from input/val.csv if available
        - Uses StratifiedKFold for cross-validation on remaining training data
        - Tokenizes text using HuggingFace AutoTokenizer
        - External validation samples are excluded from training sets
        """