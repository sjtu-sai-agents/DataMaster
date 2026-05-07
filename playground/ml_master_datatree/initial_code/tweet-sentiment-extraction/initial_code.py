import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, Subset
import torch.cuda.amp as amp
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm.auto import tqdm
import csv

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LEN = 160
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 32
EPOCHS = 2
LEARNING_RATE = 2e-5
NUM_FOLDS = 5
ADV_EPSILON = 0.2
GRAD_CLIP = 1.0
WARMUP_RATIO = 0.1
SEED = 42

# ------------------------------------------------------------
# Set seeds
# ------------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(SEED)

# ------------------------------------------------------------
# Metric
# ------------------------------------------------------------
def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    denom = len(a) + len(b) - len(c)
    return len(c) / denom if denom > 0 else 0.0

# ------------------------------------------------------------
# Span alignment utilities
# ------------------------------------------------------------
def find_span(text, selected_text):
    text = str(text).lower()
    selected = str(selected_text).lower()
    if len(selected) == 0:
        return 0, len(text)
    start = text.find(selected)
    if start != -1:
        end = start + len(selected)
        return start, end
    # try stripped version
    selected_stripped = selected.strip()
    if selected_stripped != selected:
        start = text.find(selected_stripped)
        if start != -1:
            end = start + len(selected_stripped)
            return start, end
    # fallback to full text
    return 0, len(text)

def char_to_token_span(offset_mapping, sequence_ids, char_start, char_end):
    """
    offset_mapping: list of (start, end) tuples for each token.
    sequence_ids: list of token type ids (0 for sentiment, 1 for text, None for special tokens)
    char_start, char_end: character positions (char_end exclusive)
    Returns token start and end (inclusive).
    """
    text_token_indices = [i for i, seq_id in enumerate(sequence_ids) if seq_id == 1]
    if not text_token_indices:
        return 0, 0
    first_text_token = text_token_indices[0]
    last_text_token = text_token_indices[-1]
    # start token
    start_token = None
    for i in text_token_indices:
        off_start, off_end = offset_mapping[i]
        if off_start <= char_start < off_end:
            start_token = i
            break
    if start_token is None:
        start_token = first_text_token
    # end token: token containing the last character
    last_char = char_end - 1 if char_end > 0 else 0
    end_token = None
    for i in text_token_indices:
        off_start, off_end = offset_mapping[i]
        if off_start <= last_char < off_end:
            end_token = i
            break
    if end_token is None:
        end_token = last_text_token
    return start_token, end_token

# ------------------------------------------------------------
# Data preparation
# ------------------------------------------------------------
def prepare_all_data(df, tokenizer, is_train=True):
    # tokenize all examples
    tokenized = tokenizer(
        df['sentiment'].astype(str).tolist(),
        df['text'].astype(str).tolist(),
        truncation='only_second',
        max_length=MAX_LEN,
        padding='max_length',
        return_offsets_mapping=True,
        return_tensors=None,
    )
    input_ids = tokenized['input_ids']
    attention_mask = tokenized['attention_mask']
    offset_mapping = tokenized['offset_mapping']
    features = []
    start_positions, end_positions = [], []
    for i in range(len(df)):
        seq_ids = tokenized.sequence_ids(i)
        text = df.iloc[i]['text']
        sentiment = df.iloc[i]['sentiment']
        feat = {
            'offset_mapping': offset_mapping[i],
            'sequence_ids': seq_ids,
            'text': text,
            'sentiment': sentiment,
        }
        if is_train:
            selected_text = df.iloc[i]['selected_text']
            feat['selected_text'] = selected_text
            char_start, char_end = find_span(text, selected_text)
            tok_start, tok_end = char_to_token_span(offset_mapping[i], seq_ids, char_start, char_end)
            start_positions.append(tok_start)
            end_positions.append(tok_end)
        features.append(feat)
    model_inputs = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
    }
    if is_train:
        labels = {
            'start_positions': start_positions,
            'end_positions': end_positions,
        }
        return model_inputs, labels, features
    else:
        return model_inputs, features

# ------------------------------------------------------------
# Dataset classes
# ------------------------------------------------------------
class SpanDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(val[idx], dtype=torch.long)
            for key, val in self.encodings.items()
        }
        item['start_positions'] = torch.tensor(self.labels['start_positions'][idx], dtype=torch.long)
        item['end_positions'] = torch.tensor(self.labels['end_positions'][idx], dtype=torch.long)
        return item

class InferenceDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(val[idx], dtype=torch.long)
            for key, val in self.encodings.items()
        }
        item['idx'] = torch.tensor(idx, dtype=torch.long)
        return item

# ------------------------------------------------------------
# Adversarial training (FGM)
# ------------------------------------------------------------
class FGM:
    def __init__(self, model, epsilon=0.2):
        self.model = model
        self.epsilon = epsilon
        self.backup = {}

    def attack(self, emb_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                param.data = self.backup[name]
        self.backup.clear()

# ------------------------------------------------------------
# Decoding function
# ------------------------------------------------------------
def decode_prediction(start_logits, end_logits, feature, max_answer_length=30):
    if feature['sentiment'] == 'neutral':
        return feature['text']
    offset_mapping = feature['offset_mapping']
    sequence_ids = feature['sequence_ids']
    # get text token indices
    text_token_indices = [i for i, seq_id in enumerate(sequence_ids) if seq_id == 1]
    if not text_token_indices:
        return feature['text']
    # search best span
    best_score = -1e9
    best_start = -1
    best_end = -1
    for start_idx in text_token_indices:
        if start_logits[start_idx] < -5:  # slight threshold to speed up
            continue
        for end_idx in text_token_indices:
            if end_idx < start_idx or (end_idx - start_idx + 1) > max_answer_length:
                continue
            score = start_logits[start_idx] + end_logits[end_idx]
            if score > best_score:
                best_score = score
                best_start = start_idx
                best_end = end_idx
    # fallback if no span found
    if best_start == -1 or best_end == -1:
        best_start = np.argmax(start_logits)
        best_end = np.argmax(end_logits)
        # ensure they are within text tokens
        if best_start not in text_token_indices:
            best_start = text_token_indices[0]
        if best_end not in text_token_indices:
            best_end = text_token_indices[-1]
        if best_end < best_start:
            best_end = best_start
    # character span
    char_start = offset_mapping[best_start][0]
    char_end = offset_mapping[best_end][1]
    pred_text = feature['text'][char_start:char_end]
    if len(pred_text.strip()) == 0:
        pred_text = feature['text']
    return pred_text

# ------------------------------------------------------------
# Prediction function (returns predictions and optionally logits)
# ------------------------------------------------------------
def predict(model, dataloader, device, features, return_logits=False):
    model.eval()
    start_logits_all = []
    end_logits_all = []
    indices_all = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Predicting', leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            idxs = batch['idx'].cpu().numpy()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            start_logits_all.append(outputs.start_logits.cpu().numpy())
            end_logits_all.append(outputs.end_logits.cpu().numpy())
            indices_all.append(idxs)
    start_logits = np.concatenate(start_logits_all, axis=0)
    end_logits = np.concatenate(end_logits_all, axis=0)
    indices = np.concatenate(indices_all, axis=0)
    # reorder
    order = np.argsort(indices)
    start_logits = start_logits[order]
    end_logits = end_logits[order]
    # decode
    predictions = []
    for i in range(len(start_logits)):
        pred = decode_prediction(start_logits[i], end_logits[i], features[i])
        predictions.append(pred)
    if return_logits:
        return predictions, start_logits, end_logits
    else:
        return predictions

# ------------------------------------------------------------
# Training function (one fold)
# ------------------------------------------------------------
def train_fold(model, train_loader, optimizer, scheduler, device, fold):
    fgm = FGM(model, epsilon=ADV_EPSILON)
    scaler = amp.GradScaler()
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        progress = tqdm(train_loader, desc=f'Fold {fold} Epoch {epoch+1}')
        for batch in progress:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_pos = batch['start_positions'].to(device)
            end_pos = batch['end_positions'].to(device)
            # forward with autocast
            with amp.autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                                start_positions=start_pos, end_positions=end_pos)
                loss = outputs.loss
            # backward
            scaler.scale(loss).backward()
            # adversarial attack
            fgm.attack()
            with amp.autocast():
                outputs_adv = model(input_ids=input_ids, attention_mask=attention_mask,
                                    start_positions=start_pos, end_positions=end_pos)
                loss_adv = outputs_adv.loss
            scaler.scale(loss_adv).backward()
            fgm.restore()
            # unscale for clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            progress.set_postfix({'loss': loss.item()})
        avg_loss = total_loss / len(train_loader)
        print(f'Fold {fold} Epoch {epoch+1} average loss: {avg_loss:.4f}')
    return model

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Create directories
    os.makedirs('./working', exist_ok=True)
    os.makedirs('./submission', exist_ok=True)

    # Load data
    print('Loading data...')
    train_df = pd.read_csv('./input/train.csv').fillna('')
    test_df = pd.read_csv('./input/test.csv').fillna('')

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    # Preprocess all data once
    print('Preprocessing training data...')
    train_inputs, train_labels, train_features = prepare_all_data(train_df, tokenizer, is_train=True)
    print('Preprocessing test data...')
    test_inputs, test_features = prepare_all_data(test_df, tokenizer, is_train=False)

    # Full training dataset (with labels)
    full_train_dataset = SpanDataset(train_inputs, train_labels)
    # Full inference datasets (without labels)
    train_inf_dataset = InferenceDataset(train_inputs)
    test_inf_dataset = InferenceDataset(test_inputs)

    # Prepare cross-validation
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(train_df, train_df['sentiment']))

    # OOF containers
    oof_preds = [None] * len(train_df)
    # Test logits accumulation
    test_start_logits = np.zeros((len(test_df), MAX_LEN), dtype=np.float32)
    test_end_logits = np.zeros((len(test_df), MAX_LEN), dtype=np.float32)

    # Cross-validation loop
    for fold, (train_idx, val_idx) in enumerate(splits):
        print(f'\n{"="*30}')
        print(f'Fold {fold+1}/{NUM_FOLDS}')
        print(f'{"="*30}')

        # DataLoaders
        train_loader = DataLoader(
            full_train_dataset,
            batch_size=TRAIN_BATCH_SIZE,
            sampler=SubsetRandomSampler(train_idx),
            num_workers=4,
            pin_memory=True
        )
        # Validation loader (inference style) - use sorted indices to align features
        sorted_val_idx = np.sort(val_idx)
        val_subset = Subset(train_inf_dataset, sorted_val_idx)
        val_loader = DataLoader(
            val_subset,
            batch_size=VALID_BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        val_features_sorted = [train_features[i] for i in sorted_val_idx]

        # Model
        model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)
        model.to(device)

        # Optimizer and scheduler
        num_train_steps = len(train_loader) * EPOCHS
        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(WARMUP_RATIO * num_train_steps),
            num_training_steps=num_train_steps
        )

        # Train
        model = train_fold(model, train_loader, optimizer, scheduler, device, fold+1)

        # Save model (optional)
        torch.save(model.state_dict(), f'./working/model_fold{fold+1}.pt')

        # Validation predictions
        print('Generating OOF predictions...')
        val_preds, val_start_logits, val_end_logits = predict(
            model, val_loader, device, val_features_sorted, return_logits=True
        )
        for idx, pred in zip(sorted_val_idx, val_preds):
            oof_preds[idx] = pred

        # Compute fold validation score
        fold_true = [train_features[i]['selected_text'] for i in sorted_val_idx]
        fold_jaccard = np.mean([jaccard(t, p) for t, p in zip(fold_true, val_preds)])
        print(f'Fold {fold+1} OOF Jaccard: {fold_jaccard:.5f}')

        # Test predictions (accumulate logits)
        print('Generating test predictions for ensemble...')
        test_loader = DataLoader(
            test_inf_dataset,
            batch_size=VALID_BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        _, fold_test_start, fold_test_end = predict(
            model, test_loader, device, test_features, return_logits=True
        )
        test_start_logits += fold_test_start
        test_end_logits += fold_test_end

        # Clean up
        del model, optimizer, scheduler, train_loader, val_loader, test_loader
        torch.cuda.empty_cache()

    # Average test logits
    test_start_logits /= NUM_FOLDS
    test_end_logits /= NUM_FOLDS

    # Decode final test predictions
    print('Decoding final test predictions...')
    test_preds = []
    for i in range(len(test_df)):
        pred = decode_prediction(test_start_logits[i], test_end_logits[i], test_features[i])
        test_preds.append(pred)

    # OOF score
    assert all(p is not None for p in oof_preds)
    oof_true = train_df['selected_text'].values
    oof_jaccard = np.mean([jaccard(t, p) for t, p in zip(oof_true, oof_preds)])
    print('\nOverall OOF Jaccard score:', oof_jaccard)

    # Save submission
    submission_df = pd.DataFrame({
        'textID': test_df['textID'],
        'selected_text': test_preds
    })
    submission_df.to_csv('./submission/submission.csv', index=False, quoting=csv.QUOTE_ALL)
    print('Submission saved to ./submission/submission.csv')

if __name__ == '__main__':
    main()