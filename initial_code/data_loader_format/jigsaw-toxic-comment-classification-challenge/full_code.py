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

import os
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModel, get_cosine_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import gc

# -------------------------------
# Configuration
# -------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# -------------------------------
# Model
# -------------------------------
class ToxicClassifier(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.transformer.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits

# -------------------------------
# Focal Loss
# -------------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)
        p = torch.sigmoid(inputs)
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = bce_loss * ((1 - p_t) ** self.gamma)
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# -------------------------------
# Training / Validation / Inference
# -------------------------------
def train_one_epoch(model, dataloader, optimizer, scheduler, scaler, loss_fn, device):
    model.train()
    total_loss = 0
    pbar = tqdm(dataloader, desc='Training', leave=False)
    for batch in pbar:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        with torch.cuda.amp.autocast():
            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    return total_loss / len(dataloader)

def evaluate(model, dataloader, label_cols, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating', leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            with torch.cuda.amp.autocast():
                logits = model(input_ids, attention_mask)

            preds = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    auc_scores = []
    for i in range(len(label_cols)):
        try:
            auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
        except ValueError:
            auc = np.nan
        auc_scores.append(auc)
    mean_auc = np.nanmean(auc_scores)
    return mean_auc, auc_scores, all_preds

def predict_test(model, test_loader, device):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Predicting test', leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            with torch.cuda.amp.autocast():
                logits = model(input_ids, attention_mask)
            preds = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(preds)
    return np.concatenate(all_preds, axis=0)

def parse_args():
    parser = argparse.ArgumentParser(description='Toxic Comment Classification Training')
    
    # Model parameters
    parser.add_argument('--model_name', type=str, default='microsoft/deberta-v3-base',
                        help='Pretrained model name')
    parser.add_argument('--max_len', type=int, default=192,
                        help='Maximum sequence length')
    parser.add_argument('--num_labels', type=int, default=6,
                        help='Number of labels')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--n_folds', type=int, default=5,
                        help='Number of folds for cross-validation')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                        help='Warmup ratio for scheduler')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for DataLoader')
    
    # Focal Loss parameters
    parser.add_argument('--focal_alpha', type=float, default=0.25,
                        help='Focal loss alpha parameter')
    parser.add_argument('--focal_gamma', type=float, default=2,
                        help='Focal loss gamma parameter')
    
    # Paths
    parser.add_argument('--input_dir', type=str, default='./input',
                        help='Input directory')
    parser.add_argument('--output_dir', type=str, default='./working',
                        help='Output directory for models')
    parser.add_argument('--submission_dir', type=str, default='./submission',
                        help='Submission directory')
    
    # Other
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.submission_dir, exist_ok=True)
    
    # Initialize data loader
    data_loader = MyDataLoader(
        model_name=args.model_name,
        max_len=args.max_len,
        n_folds=args.n_folds,
        random_seed=args.seed
    )
    
    # Get data
    train_data, test_data = data_loader.get_data()
    
    train_df = train_data['train_df']
    tokenizer = train_data['tokenizer']
    label_cols = train_data['label_cols']
    max_len = train_data['max_len']
    n_folds = train_data['n_folds']
    has_val_csv = train_data['has_val_csv']
    
    test_df = test_data['test_df']
    
    print(f"Training data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    print(f"Has val.csv: {has_val_csv}")
    print(f"Label columns: {label_cols}")
    
    # Prepare test dataset and loader
    test_dataset = ToxicDataset(test_df['comment_text'].values, labels=None, 
                                 tokenizer=tokenizer, max_len=max_len)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, 
                              num_workers=args.num_workers)
    
    # Arrays to hold fold predictions and validation metrics
    fold_test_preds = []
    fold_val_aucs = []
    overall_val_aucs = []
    
    # Initialize loss function
    loss_fn = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
    
    for fold in range(n_folds):
        print(f"\n====== Fold {fold+1}/{n_folds} ======")
        
        # Split train/val indices
        # If val.csv exists, exclude external validation samples from training
        if has_val_csv and 'is_external_val' in train_df.columns:
            train_idx = train_df[(train_df['fold'] != fold) & (~train_df['is_external_val'])].index
            val_idx = train_df[train_df['fold'] == fold].index
        else:
            train_idx = train_df[train_df['fold'] != fold].index
            val_idx = train_df[train_df['fold'] == fold].index

        train_texts = train_df.loc[train_idx, 'comment_text'].values
        val_texts = train_df.loc[val_idx, 'comment_text'].values

        train_labels = train_df.loc[train_idx, label_cols].values
        val_labels = train_df.loc[val_idx, label_cols].values

        print(f"Train samples: {len(train_texts)}, Val samples: {len(val_texts)}")

        # Datasets and DataLoaders
        train_dataset = ToxicDataset(train_texts, train_labels, tokenizer=tokenizer, max_len=max_len)
        val_dataset = ToxicDataset(val_texts, val_labels, tokenizer=tokenizer, max_len=max_len)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                                   num_workers=args.num_workers, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                                 num_workers=args.num_workers, pin_memory=True)

        # Model, optimizer, loss, scheduler, scaler
        model = ToxicClassifier(args.model_name, len(label_cols)).to(device)
        optimizer = AdamW(model.parameters(), lr=args.lr)
        total_steps = len(train_loader) * args.epochs
        num_warmup_steps = int(args.warmup_ratio * total_steps)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, total_steps)
        scaler = torch.cuda.amp.GradScaler()

        best_val_auc = 0.0
        best_epoch = 0
        model_path = os.path.join(args.output_dir, f'model_fold{fold}.pt')

        for epoch in range(args.epochs):
            print(f"Epoch {epoch+1}/{args.epochs}")
            train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, loss_fn, device)
            val_auc, val_aucs, _ = evaluate(model, val_loader, label_cols, device)
            print(f"Train Loss: {train_loss:.4f} - Val Mean AUC: {val_auc:.4f}")
            print("Per-class Val AUC:", dict(zip(label_cols, [f"{v:.4f}" for v in val_aucs])))
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_epoch = epoch
                torch.save(model.state_dict(), model_path)

        print(f"Fold {fold+1} best epoch {best_epoch+1} with mean AUC {best_val_auc:.4f}")
        fold_val_aucs.append(best_val_auc)
        overall_val_aucs.append(best_val_auc)

        # Load best model and predict on test set
        model.load_state_dict(torch.load(model_path))
        test_preds = predict_test(model, test_loader, device)
        fold_test_preds.append(test_preds)

        # Clean up
        del model, train_loader, val_loader, train_dataset, val_dataset, optimizer, scheduler, scaler
        gc.collect()
        torch.cuda.empty_cache()

    # -------------------------------
    # Ensemble predictions and create submission
    # -------------------------------
    ensemble_test_preds = np.mean(fold_test_preds, axis=0)

    submission = pd.DataFrame({
        'id': test_df['id']
    })
    submission[label_cols] = ensemble_test_preds
    submission.to_csv(os.path.join(args.submission_dir, 'submission.csv'), index=False)
    print(f"\nSubmission saved to {os.path.join(args.submission_dir, 'submission.csv')}")

    # -------------------------------
    # Print overall validation metric
    # -------------------------------
    print("\nOverall cross‑validation metrics:")
    for fold, auc in enumerate(overall_val_aucs):
        print(f"Fold {fold+1}: Mean AUC = {auc:.4f}")
    mean_cv_auc = np.mean(overall_val_aucs)
    std_cv_auc = np.std(overall_val_aucs)
    print(f"Mean CV AUC: {mean_cv_auc:.4f} ± {std_cv_auc:.4f}")

if __name__ == "__main__":
    main()