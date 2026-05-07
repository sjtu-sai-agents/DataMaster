import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import RobertaTokenizer


def clean_text(t):
    """Clean text by stripping quotes and decoding unicode escapes."""
    if pd.isna(t):
        return ""
    t = t.strip('"')
    try:
        t = t.encode("utf-8").decode("unicode_escape", errors="ignore")
    except:
        pass
    return t


class InsultDataset(Dataset):
    """Dataset class for insult detection."""
    
    def __init__(self, texts, labels=None, tokenizer=None, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        else:
            item['labels'] = torch.tensor(0.0, dtype=torch.float)
        return item


class MyDataLoader(BaseDataLoader):
    """Data loader for insult detection task."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = kwargs.get('model_name', 'roberta-large')
        self.max_len = kwargs.get('max_len', 128)
        self.tokenizer = None
        self.has_fixed_val = False
        
    def setup(self):
        """
        Load data, clean text, and prepare datasets.
        Checks for fixed validation set in input/val.csv.
        """
        # Load tokenizer
        self.tokenizer = RobertaTokenizer.from_pretrained(self.model_name)
        
        # Load data
        train_df = pd.read_csv('input/train.csv')
        test_df = pd.read_csv('input/test.csv')
        
        # Clean text
        train_df["comment_clean"] = train_df["Comment"].apply(clean_text)
        test_df["comment_clean"] = test_df["Comment"].apply(clean_text)
        
        # Check for fixed validation set
        val_df = None
        if os.path.exists('input/val.csv'):
            val_df = pd.read_csv('input/val.csv')
            if len(val_df) > 0 and 'Comment' in val_df.columns:
                val_df["comment_clean"] = val_df["Comment"].apply(clean_text)
                # Remove validation samples from training data
                val_comments = set(val_df['Comment'].values)
                train_df = train_df[~train_df['Comment'].isin(val_comments)]
                self.has_fixed_val = True
            else:
                val_df = None
        
        # Store data
        self.train_data = {
            'train_df': train_df,
            'val_df': val_df,
            'labels': train_df["Insult"].values if 'Insult' in train_df.columns else None
        }
        self.test_data = {
            'test_df': test_df
        }
        
    def describe(self) -> str:
        """Return description of data processing approach."""
        desc = "Data loader for insult detection using RoBERTa. "
        desc += "Includes text cleaning (quote stripping, unicode escape decoding). "
        if self.has_fixed_val:
            desc += "Uses fixed validation set from input/val.csv."
        else:
            desc += "No fixed validation set; cross-validation will be used."
        return desc