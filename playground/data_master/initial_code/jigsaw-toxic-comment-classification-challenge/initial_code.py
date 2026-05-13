import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import gc

# Set random seeds for reproducibility
RANDOM_SEED = 42
def set_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed()

# -------------------------------
# Configuration
# -------------------------------
MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LEN = 192
BATCH_SIZE = 16
EPOCHS = 3
LR = 2e-5
N_FOLDS = 5
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2
WARMUP_RATIO = 0.1
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = 4

# -------------------------------
# Data Loading
# -------------------------------
train_df = pd.read_csv('./input/train.csv')
test_df = pd.read_csv('./input/test.csv')
label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

# Add fold column using stratified kfold on sum of labels
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
train_df['fold'] = -1
for fold, (_, val_idx) in enumerate(skf.split(train_df, train_df[label_cols].sum(axis=1))):
    train_df.loc[val_idx, 'fold'] = fold

# -------------------------------
# Tokenizer
# -------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# -------------------------------
# Dataset
# -------------------------------
class ToxicDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=tokenizer, max_len=MAX_LEN):
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
# Model
# -------------------------------
class ToxicClassifier(nn.Module):
    def __init__(self, model_name=MODEL_NAME):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.transformer.config.hidden_size, len(label_cols))

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
    def __init__(self, alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA, reduction='mean'):
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
def train_one_epoch(model, dataloader, optimizer, scheduler, scaler):
    model.train()
    total_loss = 0
    pbar = tqdm(dataloader, desc='Training', leave=False)
    for batch in pbar:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)

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

def evaluate(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating', leave=False):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

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

def predict_test(model, test_loader):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Predicting test', leave=False):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            with torch.cuda.amp.autocast():
                logits = model(input_ids, attention_mask)
            preds = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(preds)
    return np.concatenate(all_preds, axis=0)

# -------------------------------
# Main Cross-Validation Loop
# -------------------------------
# Prepare test dataset and loader
test_dataset = ToxicDataset(test_df['comment_text'].values, labels=None)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# Arrays to hold fold predictions and validation metrics
fold_test_preds = []
fold_val_aucs = []
overall_val_aucs = []  # per fold mean AUC

for fold in range(N_FOLDS):
    print(f"\n====== Fold {fold+1}/{N_FOLDS} ======")
    # Split train/val indices
    train_idx = train_df[train_df['fold'] != fold].index
    val_idx = train_df[train_df['fold'] == fold].index

    train_texts = train_df.loc[train_idx, 'comment_text'].values
    val_texts = train_df.loc[val_idx, 'comment_text'].values

    train_labels = train_df.loc[train_idx, label_cols].values
    val_labels = train_df.loc[val_idx, label_cols].values

    # Datasets and DataLoaders
    train_dataset = ToxicDataset(train_texts, train_labels)
    val_dataset = ToxicDataset(val_texts, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # Model, optimizer, loss, scheduler, scaler
    model = ToxicClassifier().to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR)
    loss_fn = FocalLoss()
    total_steps = len(train_loader) * EPOCHS
    num_warmup_steps = int(WARMUP_RATIO * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, total_steps)
    scaler = torch.cuda.amp.GradScaler()

    best_val_auc = 0.0
    best_epoch = 0
    # Store best model predictions on validation? We'll just track model weights
    model_path = f'./working/model_fold{fold}.pt'

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, scaler)
        val_auc, val_aucs, _ = evaluate(model, val_loader)
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
    test_preds = predict_test(model, test_loader)
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
submission.to_csv('./submission/submission.csv', index=False)
print("\nSubmission saved to ./submission/submission.csv")

# -------------------------------
# Print overall validation metric
# -------------------------------
print("\nOverall cross‑validation metrics:")
for fold, auc in enumerate(overall_val_aucs):
    print(f"Fold {fold+1}: Mean AUC = {auc:.4f}")
mean_cv_auc = np.mean(overall_val_aucs)
std_cv_auc = np.std(overall_val_aucs)
print(f"Mean CV AUC: {mean_cv_auc:.4f} ± {std_cv_auc:.4f}")