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

import os
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import RobertaForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


def set_seed(seed):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train RoBERTa for insult detection')
    
    # Model parameters
    parser.add_argument('--model_name', type=str, default='roberta-large',
                        help='Pretrained model name')
    parser.add_argument('--max_len', type=int, default=128,
                        help='Maximum sequence length')
    
    # Training parameters
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=3,
                        help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--early_stop_patience', type=int, default=2,
                        help='Early stopping patience')
    parser.add_argument('--num_folds', type=int, default=5,
                        help='Number of folds for cross-validation')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of data loader workers')
    
    # Path parameters
    parser.add_argument('--input_dir', type=str, default='./input',
                        help='Input directory')
    parser.add_argument('--output_dir', type=str, default='./submission',
                        help='Output directory')
    
    return parser.parse_args()


def train_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, device):
    """Train for one epoch."""
    model.train()
    train_loss = 0
    progress_bar = tqdm(train_loader, desc="Training")
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits.squeeze(-1)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        train_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
    
    return train_loss / len(train_loader)


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    val_preds = []
    val_true = []
    val_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            with torch.cuda.amp.autocast():
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits.squeeze(-1)
                loss = criterion(logits, labels)
            val_loss += loss.item()
            probs = torch.sigmoid(logits).cpu().numpy()
            val_preds.extend(probs)
            val_true.extend(labels.cpu().numpy())
    
    val_auc = roc_auc_score(val_true, val_preds)
    return val_loss / len(val_loader), val_auc, np.array(val_preds)


def predict(model, data_loader, device):
    """Make predictions."""
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Prediction"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            with torch.cuda.amp.autocast():
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits.squeeze(-1)
                probs = torch.sigmoid(logits).cpu().numpy()
                predictions.extend(probs)
    return np.array(predictions)


def train_single_split(train_texts, train_labels, val_texts, val_labels, 
                       test_texts, tokenizer, args, device):
    """Train with single train/val split."""
    # Class weight for imbalance
    pos_weight = torch.tensor(
        [(len(train_labels) - sum(train_labels)) / (sum(train_labels) + 1e-5)], 
        device=device
    )
    
    # Datasets and DataLoaders
    train_dataset = InsultDataset(train_texts, train_labels, tokenizer, args.max_len)
    val_dataset = InsultDataset(val_texts, val_labels, tokenizer, args.max_len)
    test_dataset = InsultDataset(test_texts, None, tokenizer, args.max_len)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                              shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                           shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                            shuffle=False, num_workers=args.num_workers)
    
    # Model
    model = RobertaForSequenceClassification.from_pretrained(args.model_name, num_labels=1)
    model.to(device)
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )
    
    # Loss
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler()
    
    best_auc = 0.0
    patience_counter = 0
    best_model_path = "model_best.bin"
    
    # Training loop
    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch+1}/{args.num_epochs}")
        avg_train_loss = train_epoch(model, train_loader, criterion, optimizer, 
                                     scheduler, scaler, device)
        print(f"Average training loss: {avg_train_loss:.4f}")
        
        avg_val_loss, val_auc, _ = validate(model, val_loader, criterion, device)
        print(f"Validation Loss: {avg_val_loss:.4f}, AUC: {val_auc:.4f}")
        
        # Early stopping check
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved with AUC: {best_auc:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement in AUC. Patience {patience_counter}/{args.early_stop_patience}")
            if patience_counter >= args.early_stop_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    model.load_state_dict(torch.load(best_model_path))
    
    # Test predictions
    test_preds = predict(model, test_loader, device)
    
    # Cleanup
    if os.path.exists(best_model_path):
        os.remove(best_model_path)
    
    print(f"Best Validation AUC: {best_auc:.4f}")
    
    return test_preds, best_auc


def train_cross_validation(train_df, y, test_texts, tokenizer, args, device):
    """Train with cross-validation."""
    skf = StratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)
    
    oof_preds = np.zeros(len(train_df))
    test_preds = np.zeros(len(test_texts))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, y)):
        print(f"\nFold {fold+1}/{args.num_folds}")
        
        # Split data
        train_texts = train_df.iloc[train_idx]["comment_clean"].values
        train_labels = y[train_idx]
        val_texts = train_df.iloc[val_idx]["comment_clean"].values
        val_labels = y[val_idx]
        
        # Class weight for imbalance
        pos_weight = torch.tensor(
            [(len(train_labels) - sum(train_labels)) / (sum(train_labels) + 1e-5)], 
            device=device
        )
        
        # Datasets and DataLoaders
        train_dataset = InsultDataset(train_texts, train_labels, tokenizer, args.max_len)
        val_dataset = InsultDataset(val_texts, val_labels, tokenizer, args.max_len)
        test_dataset = InsultDataset(test_texts, None, tokenizer, args.max_len)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                                  shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                               shuffle=False, num_workers=args.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                                shuffle=False, num_workers=args.num_workers)
        
        # Model
        model = RobertaForSequenceClassification.from_pretrained(args.model_name, num_labels=1)
        model.to(device)
        
        # Optimizer and scheduler
        optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        total_steps = len(train_loader) * args.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=total_steps
        )
        
        # Loss
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        # Mixed precision scaler
        scaler = torch.cuda.amp.GradScaler()
        
        best_auc = 0.0
        patience_counter = 0
        best_model_path = f"model_fold{fold}.bin"
        
        # Training loop
        for epoch in range(args.num_epochs):
            print(f"Epoch {epoch+1}/{args.num_epochs}")
            avg_train_loss = train_epoch(model, train_loader, criterion, optimizer, 
                                         scheduler, scaler, device)
            print(f"Average training loss: {avg_train_loss:.4f}")
            
            avg_val_loss, val_auc, _ = validate(model, val_loader, criterion, device)
            print(f"Validation Loss: {avg_val_loss:.4f}, AUC: {val_auc:.4f}")
            
            # Early stopping check
            if val_auc > best_auc:
                best_auc = val_auc
                patience_counter = 0
                torch.save(model.state_dict(), best_model_path)
                print(f"Best model saved with AUC: {best_auc:.4f}")
            else:
                patience_counter += 1
                print(f"No improvement in AUC. Patience {patience_counter}/{args.early_stop_patience}")
                if patience_counter >= args.early_stop_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model for this fold
        model.load_state_dict(torch.load(best_model_path))
        
        # Out-of-fold predictions
        _, _, val_preds = validate(model, val_loader, criterion, device)
        oof_preds[val_idx] = val_preds
        
        # Test predictions for this fold
        fold_test_preds = predict(model, test_loader, device)
        test_preds += fold_test_preds / args.num_folds
        
        # Cleanup
        del model, optimizer, scheduler, train_loader, val_loader, test_loader
        del train_dataset, val_dataset, test_dataset
        torch.cuda.empty_cache()
        if os.path.exists(best_model_path):
            os.remove(best_model_path)
    
    # Overall OOF AUC
    oof_auc = roc_auc_score(y, oof_preds)
    print(f"\nOverall OOF AUC: {oof_auc:.4f}")
    
    return test_preds, oof_auc


def main():
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data using MyDataLoader
    data_loader = MyDataLoader(
        model_name=args.model_name,
        max_len=args.max_len
    )
    train_data, test_data = data_loader.get_data()
    
    train_df = train_data['train_df']
    val_df = train_data['val_df']
    y = train_data['labels']
    test_df = test_data['test_df']
    tokenizer = data_loader.tokenizer
    
    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # Train model
    if val_df is not None:
        # Use fixed validation set
        print("Using fixed validation set from val.csv")
        print(f"Validation samples: {len(val_df)}")
        test_preds, best_auc = train_single_split(
            train_df["comment_clean"].values,
            y,
            val_df["comment_clean"].values,
            val_df["Insult"].values,
            test_df["comment_clean"].values,
            tokenizer,
            args,
            device
        )
    else:
        # Use cross-validation
        print("Using cross-validation (no val.csv found)")
        test_preds, oof_auc = train_cross_validation(
            train_df, y, test_df["comment_clean"].values, tokenizer, args, device
        )
    
    # Create submission file
    submission_df = test_df[["Comment", "Date"]].copy()
    submission_df["Insult"] = test_preds
    submission_path = os.path.join(args.output_dir, "submission.csv")
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")


if __name__ == "__main__":
    main()