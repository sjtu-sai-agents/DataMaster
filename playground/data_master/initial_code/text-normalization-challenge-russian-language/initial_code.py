import os
import gc
import re
import time
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import GroupKFold
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
VOCAB_SIZE = 120000
SUFFIX_VOCAB_SIZE = 15000
PREFIX_VOCAB_SIZE = 7500
AMBIGUITY_THRESHOLD = 0.97
MAX_VARIANTS = 50
WINDOW_SIZE = 6
SEQ_LEN = 2 * WINDOW_SIZE + 1
BATCH_SIZE = 2048
EPOCHS = 15
N_FOLDS = 5
LR = 0.001
WORD_EMB_DIM = 96
SUFFIX_EMB_DIM = 48
PREFIX_EMB_DIM = 48
CASE_EMB_DIM = 16
CHAR_EMB_DIM = 32
CHAR_OUT_DIM = 128
D_MODEL = 512
NHEAD = 8
NUM_LAYERS = 6
FF_DIM = 2048
DROPOUT = 0.1
LABEL_SMOOTHING = 0.1
CLASS_LOSS_WEIGHT = 0.5
MAX_CHAR_LEN = 20

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# ----------------------------------------------------------------------
# Data Loading
# ----------------------------------------------------------------------
print('Loading data...')
train = pd.read_csv(
    './input/ru_train.csv',
    dtype={
        'sentence_id': 'int32',
        'token_id': 'int16',
        'class': 'category',
        'before': 'object',
        'after': 'object'
    },
    keep_default_na=False
)

test = pd.read_csv(
    './input/ru_test_2.csv',
    dtype={
        'sentence_id': 'int32',
        'token_id': 'int16',
        'before': 'object'
    },
    keep_default_na=False
)

# ----------------------------------------------------------------------
# Token statistics and ambiguous set
# ----------------------------------------------------------------------
print('Building token statistics...')
token_counter = Counter(zip(train['before'], train['after']))
token_stats = {}
for (b, a), cnt in token_counter.items():
    token_stats.setdefault(b, []).append((a, cnt))

for b in token_stats:
    token_stats[b].sort(key=lambda x: x[1], reverse=True)

token_map = {}
ambiguous_tokens = set()
for b, lst in token_stats.items():
    total = sum(cnt for _, cnt in lst)
    top_cnt = lst[0][1]
    token_map[b] = [a for a, _ in lst[:MAX_VARIANTS]]
    if top_cnt / total < AMBIGUITY_THRESHOLD:
        ambiguous_tokens.add(b)

# ----------------------------------------------------------------------
# Vocabularies
# ----------------------------------------------------------------------
print('Building vocabularies...')
# Word vocabulary (most frequent before tokens)
before_counts = train['before'].value_counts()
top_before = before_counts.index[:VOCAB_SIZE].tolist()
word2idx = {w: i+1 for i, w in enumerate(top_before)}  # 0 = unknown

# Suffix vocabulary (last 3 chars)
unique_train_before = train['before'].unique()
suffix_counter = Counter()
for token in unique_train_before:
    if len(token) >= 3:
        suffix_counter[token[-3:]] += 1
top_suffixes = [suf for suf, _ in suffix_counter.most_common(SUFFIX_VOCAB_SIZE)]
suffix2idx = {s: i+1 for i, s in enumerate(top_suffixes)}

# Prefix vocabulary (first 3 chars)
prefix_counter = Counter()
for token in unique_train_before:
    if len(token) >= 3:
        prefix_counter[token[:3]] += 1
top_prefixes = [pre for pre, _ in prefix_counter.most_common(PREFIX_VOCAB_SIZE)]
prefix2idx = {p: i+1 for i, p in enumerate(top_prefixes)}

# Casing categories
def get_case_id(token):
    if token.islower():
        return 1
    elif token.istitle():
        return 2
    elif token.isupper():
        return 3
    else:
        return 4

# Character vocabulary (from all tokens in train+test)
all_tokens = set(train['before'].unique()) | set(test['before'].unique())
all_chars = set()
for token in all_tokens:
    all_chars.update(token)
char_list = sorted(all_chars)
char2idx = {ch: i+1 for i, ch in enumerate(char_list)}  # 0 = padding/unknown

# Unique token mapping (for character matrix)
all_unique_tokens = list(all_tokens)
u_token2idx = {t: i+1 for i, t in enumerate(all_unique_tokens)}  # 0 = unknown (should not occur)

# Character matrix: shape (num_u_tokens+1, MAX_CHAR_LEN)
print('Building character matrix...')
num_u_tokens = len(all_unique_tokens)
char_matrix = np.zeros((num_u_tokens+1, MAX_CHAR_LEN), dtype=np.int16)
for token, idx in tqdm(u_token2idx.items()):
    for j, ch in enumerate(token[:MAX_CHAR_LEN]):
        char_matrix[idx, j] = char2idx.get(ch, 0)

# ----------------------------------------------------------------------
# Feature mapping on raw data
# ----------------------------------------------------------------------
print('Mapping features...')
def map_word(token):
    return word2idx.get(token, 0)

def map_suffix(token):
    if len(token) >= 3:
        return suffix2idx.get(token[-3:], 0)
    return 0

def map_prefix(token):
    if len(token) >= 3:
        return prefix2idx.get(token[:3], 0)
    return 0

def map_case(token):
    return get_case_id(token)

def map_u(token):
    return u_token2idx.get(token, 0)

train['w_idx'] = train['before'].map(map_word).astype('int32')
train['s_idx'] = train['before'].map(map_suffix).astype('int16')
train['p_idx'] = train['before'].map(map_prefix).astype('int16')
train['c_idx'] = train['before'].map(map_case).astype('int8')
train['u_idx'] = train['before'].map(map_u).astype('int32')

test['w_idx'] = test['before'].map(map_word).astype('int32')
test['s_idx'] = test['before'].map(map_suffix).astype('int16')
test['p_idx'] = test['before'].map(map_prefix).astype('int16')
test['c_idx'] = test['before'].map(map_case).astype('int8')
test['u_idx'] = test['before'].map(map_u).astype('int32')

# ----------------------------------------------------------------------
# Context window features
# ----------------------------------------------------------------------
def add_context_features(df, window):
    """Add left and right context columns for each feature."""
    df = df.sort_values(['sentence_id', 'token_id']).reset_index(drop=True)
    feat_cols = ['w_idx', 's_idx', 'p_idx', 'c_idx', 'u_idx']
    dtypes = [np.int32, np.int16, np.int16, np.int8, np.int32]
    for offset in range(1, window+1):
        for col, dtype in zip(feat_cols, dtypes):
            # Left
            left = df.groupby('sentence_id')[col].shift(offset).fillna(0).astype(dtype)
            df[f'L{offset}_{col}'] = left
            # Right
            right = df.groupby('sentence_id')[col].shift(-offset).fillna(0).astype(dtype)
            df[f'R{offset}_{col}'] = right
    # Rename center columns
    rename_dict = {col: f'C_{col}' for col in feat_cols}
    df.rename(columns=rename_dict, inplace=True)
    return df

print('Adding context to training data...')
train = add_context_features(train, WINDOW_SIZE)
print('Adding context to test data...')
test = add_context_features(test, WINDOW_SIZE)

# ----------------------------------------------------------------------
# Ambiguous flags and class encoding
# ----------------------------------------------------------------------
train['is_ambiguous'] = train['before'].isin(ambiguous_tokens)
test['is_ambiguous'] = test['before'].isin(ambiguous_tokens)

# Class encoding
class_cats = train['class'].astype('category')
train['class_code'] = class_cats.cat.codes.astype('int8')
num_classes = len(class_cats.cat.categories)

# ----------------------------------------------------------------------
# Prepare ambiguous training data
# ----------------------------------------------------------------------
print('Preparing ambiguous training data...')
ambig_df = train[train['is_ambiguous']].copy()

# Target index within top candidates
def get_target_idx(row):
    b = row['before']
    a = row['after']
    cand = token_map.get(b, [])
    try:
        return cand.index(a)
    except ValueError:
        return 0

ambig_df['target'] = ambig_df.apply(get_target_idx, axis=1).astype('int8')

# Extract sequence arrays
pos_order = [f'L{i}' for i in range(WINDOW_SIZE, 0, -1)] + ['C'] + [f'R{i}' for i in range(1, WINDOW_SIZE+1)]

word_cols = [f'{pos}_w_idx' for pos in pos_order]
suffix_cols = [f'{pos}_s_idx' for pos in pos_order]
prefix_cols = [f'{pos}_p_idx' for pos in pos_order]
case_cols = [f'{pos}_c_idx' for pos in pos_order]
uidx_cols = [f'{pos}_u_idx' for pos in pos_order]

X_word = ambig_df[word_cols].values.astype('int32')
X_suffix = ambig_df[suffix_cols].values.astype('int16')
X_prefix = ambig_df[prefix_cols].values.astype('int16')
X_case = ambig_df[case_cols].values.astype('int8')
X_uidx = ambig_df[uidx_cols].values.astype('int32')
y_norm = ambig_df['target'].values.astype('int64')
y_class = ambig_df['class_code'].values.astype('int64')
sentence_ids = ambig_df['sentence_id'].values

# Free memory
del train, ambig_df
gc.collect()

# ----------------------------------------------------------------------
# Dataset and DataLoader
# ----------------------------------------------------------------------
class AmbiguousDataset(Dataset):
    def __init__(self, word, suffix, prefix, case, uidx, norm_target=None, class_target=None):
        self.word = torch.from_numpy(word)
        self.suffix = torch.from_numpy(suffix)
        self.prefix = torch.from_numpy(prefix)
        self.case = torch.from_numpy(case)
        self.uidx = torch.from_numpy(uidx)
        self.has_target = norm_target is not None
        if self.has_target:
            self.norm_target = torch.from_numpy(norm_target)
            self.class_target = torch.from_numpy(class_target)

    def __len__(self):
        return len(self.word)

    def __getitem__(self, idx):
        if self.has_target:
            return (self.word[idx], self.suffix[idx], self.prefix[idx],
                    self.case[idx], self.uidx[idx],
                    self.norm_target[idx], self.class_target[idx])
        else:
            return (self.word[idx], self.suffix[idx], self.prefix[idx],
                    self.case[idx], self.uidx[idx])

# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1, reduction='mean'):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, logits, target):
        log_probs = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            n_classes = logits.size(-1)
            smooth_target = torch.ones_like(log_probs) * (self.smoothing / (n_classes - 1))
            smooth_target.scatter_(1, target.unsqueeze(1), 1 - self.smoothing)
        loss = - (smooth_target * log_probs).sum(dim=-1)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

class TransformerDeepCharModel(nn.Module):
    def __init__(self, word_vocab_size, suffix_vocab_size, prefix_vocab_size,
                 case_vocab_size, char_vocab_size, char_out_dim, d_model,
                 nhead, num_layers, ff_dim, dropout, max_variants, num_classes,
                 max_char_len, window_size):
        super().__init__()
        self.window_size = window_size
        self.seq_len = 2 * window_size + 1

        self.word_emb = nn.Embedding(word_vocab_size + 1, WORD_EMB_DIM, padding_idx=0)
        self.suffix_emb = nn.Embedding(suffix_vocab_size + 1, SUFFIX_EMB_DIM, padding_idx=0)
        self.prefix_emb = nn.Embedding(prefix_vocab_size + 1, PREFIX_EMB_DIM, padding_idx=0)
        self.case_emb = nn.Embedding(case_vocab_size + 1, CASE_EMB_DIM, padding_idx=0)  # case_vocab_size = 4

        self.char_embed = nn.Embedding(char_vocab_size + 1, CHAR_EMB_DIM, padding_idx=0)
        self.char_conv1 = nn.Conv1d(CHAR_EMB_DIM, 64, kernel_size=3, padding=1)
        self.char_conv2 = nn.Conv1d(64, char_out_dim, kernel_size=3, padding=1)
        self.char_pool = nn.AdaptiveMaxPool1d(1)
        self.char_act = nn.GELU()
        self.char_dropout = nn.Dropout(dropout)

        total_emb_dim = WORD_EMB_DIM + SUFFIX_EMB_DIM + PREFIX_EMB_DIM + CASE_EMB_DIM + char_out_dim
        self.proj = nn.Linear(total_emb_dim, d_model)

        self.pos_embed = nn.Embedding(self.seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ff_dim,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.dropout = nn.Dropout(dropout)

        self.center_idx = window_size
        combined_dim = d_model * 3
        self.norm_head = nn.Linear(combined_dim, max_variants)
        self.class_head = nn.Linear(combined_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, word_seq, suffix_seq, prefix_seq, case_seq, uidx_seq, char_matrix):
        # Ensure indices are long tensors for embedding layers
        word_seq = word_seq.long()
        suffix_seq = suffix_seq.long()
        prefix_seq = prefix_seq.long()
        case_seq = case_seq.long()
        uidx_seq = uidx_seq.long()

        batch_size, seq_len = word_seq.size()
        device = word_seq.device

        word_emb = self.word_emb(word_seq)
        suffix_emb = self.suffix_emb(suffix_seq)
        prefix_emb = self.prefix_emb(prefix_seq)
        case_emb = self.case_emb(case_seq)

        # Character features
        flat_uidx = uidx_seq.view(-1)
        char_indices = char_matrix[flat_uidx]  # (batch*seq_len, MAX_CHAR_LEN)
        char_emb = self.char_embed(char_indices)          # (B*S, L, E)
        char_emb = char_emb.permute(0, 2, 1)              # (B*S, E, L)
        char_feat = self.char_act(self.char_conv1(char_emb))
        char_feat = self.char_act(self.char_conv2(char_feat))
        char_feat = self.char_pool(char_feat).squeeze(-1) # (B*S, char_out_dim)
        char_feat = self.char_dropout(char_feat)
        char_feat = char_feat.view(batch_size, seq_len, -1)

        combined = torch.cat([word_emb, suffix_emb, prefix_emb, case_emb, char_feat], dim=2)
        combined = self.proj(combined)

        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_embed(positions)
        x = combined + pos_emb
        x = self.dropout(x)

        x = self.transformer(x)

        center = x[:, self.center_idx, :]
        avg_pool = torch.mean(x, dim=1)
        max_pool = torch.max(x, dim=1)[0]
        agg = torch.cat([center, avg_pool, max_pool], dim=1)
        agg = self.dropout(agg)

        norm_logits = self.norm_head(agg)
        class_logits = self.class_head(agg)
        return norm_logits, class_logits

# ----------------------------------------------------------------------
# Training and evaluation functions
# ----------------------------------------------------------------------
def train_epoch(model, loader, optimizer, scheduler,
                norm_criterion, class_criterion, device, char_matrix):
    model.train()
    total_loss = 0.0
    correct_norm = 0
    total = 0
    for batch in tqdm(loader, desc='Training', leave=False):
        word, suffix, prefix, case, uidx, norm_target, class_target = [x.to(device) for x in batch]
        optimizer.zero_grad()
        norm_logits, class_logits = model(word, suffix, prefix, case, uidx, char_matrix)
        loss_norm = norm_criterion(norm_logits, norm_target)
        loss_class = class_criterion(class_logits, class_target)
        loss = loss_norm + CLASS_LOSS_WEIGHT * loss_class
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * word.size(0)
        pred_norm = norm_logits.argmax(dim=1)
        correct_norm += (pred_norm == norm_target).sum().item()
        total += word.size(0)

    avg_loss = total_loss / total if total else 0
    accuracy = correct_norm / total if total else 0
    return avg_loss, accuracy

def evaluate(model, loader, device, char_matrix):
    model.eval()
    correct_norm = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc='Evaluating', leave=False):
            word, suffix, prefix, case, uidx, norm_target, class_target = [x.to(device) for x in batch]
            norm_logits, _ = model(word, suffix, prefix, case, uidx, char_matrix)
            pred_norm = norm_logits.argmax(dim=1)
            correct_norm += (pred_norm == norm_target).sum().item()
            total += word.size(0)
    accuracy = correct_norm / total if total else 0
    return accuracy

# ----------------------------------------------------------------------
# Cross‑validation training
# ----------------------------------------------------------------------
# Move character matrix to device once
char_matrix_tensor = torch.from_numpy(char_matrix).long().to(device)

# Prepare data for CV
X = {
    'word': X_word,
    'suffix': X_suffix,
    'prefix': X_prefix,
    'case': X_case,
    'uidx': X_uidx,
    'norm': y_norm,
    'class': y_class
}
groups = sentence_ids

kf = GroupKFold(n_splits=N_FOLDS)

fold_val_accs = []
models = []  # store trained models for ensemble

for fold, (train_idx, val_idx) in enumerate(kf.split(X['word'], groups=groups)):
    print(f'\n--- Fold {fold+1}/{N_FOLDS} ---')

    # Datasets
    train_dataset = AmbiguousDataset(
        X['word'][train_idx], X['suffix'][train_idx], X['prefix'][train_idx],
        X['case'][train_idx], X['uidx'][train_idx],
        X['norm'][train_idx], X['class'][train_idx]
    )
    val_dataset = AmbiguousDataset(
        X['word'][val_idx], X['suffix'][val_idx], X['prefix'][val_idx],
        X['case'][val_idx], X['uidx'][val_idx],
        X['norm'][val_idx], X['class'][val_idx]
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True
    )

    # Model, optimizer, criterion
    model = TransformerDeepCharModel(
        word_vocab_size=VOCAB_SIZE,
        suffix_vocab_size=SUFFIX_VOCAB_SIZE,
        prefix_vocab_size=PREFIX_VOCAB_SIZE,
        case_vocab_size=4,
        char_vocab_size=len(char2idx),
        char_out_dim=CHAR_OUT_DIM,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        ff_dim=FF_DIM,
        dropout=DROPOUT,
        max_variants=MAX_VARIANTS,
        num_classes=num_classes,
        max_char_len=MAX_CHAR_LEN,
        window_size=WINDOW_SIZE
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    total_steps = len(train_loader) * EPOCHS
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, total_steps=total_steps, pct_start=0.1
    )
    norm_criterion = LabelSmoothingCrossEntropy(smoothing=LABEL_SMOOTHING)
    class_criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, EPOCHS+1):
        start = time.time()
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler,
            norm_criterion, class_criterion, device, char_matrix_tensor
        )
        val_acc = evaluate(model, val_loader, device, char_matrix_tensor)
        elapsed = time.time() - start
        print(f'Epoch {epoch:2d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Time: {elapsed:.0f}s')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict().copy()

    print(f'Fold {fold+1} best validation accuracy: {best_val_acc:.4f}')
    fold_val_accs.append(best_val_acc)

    # Save best model of this fold
    model.load_state_dict(best_state)
    models.append(model)

# Print CV results
print('\nCross‑validation results:')
for fold, acc in enumerate(fold_val_accs):
    print(f'Fold {fold+1}: {acc:.4f}')
print(f'Average: {np.mean(fold_val_accs):.4f}')

# ----------------------------------------------------------------------
# Inference on test set
# ----------------------------------------------------------------------
print('\nPreparing test data...')
test_ambig = test[test['is_ambiguous']].copy()
test_nonambig = test[~test['is_ambiguous']].copy()  # keep for later

# Extract arrays for ambiguous test tokens
test_word_seq = test_ambig[word_cols].values.astype('int32')
test_suffix_seq = test_ambig[suffix_cols].values.astype('int16')
test_prefix_seq = test_ambig[prefix_cols].values.astype('int16')
test_case_seq = test_ambig[case_cols].values.astype('int8')
test_uidx_seq = test_ambig[uidx_cols].values.astype('int32')

test_ambig_dataset = AmbiguousDataset(
    test_word_seq, test_suffix_seq, test_prefix_seq,
    test_case_seq, test_uidx_seq
)
test_ambig_loader = DataLoader(
    test_ambig_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=4, pin_memory=True
)

# Accumulate probabilities from all folds
all_probs = np.zeros((len(test_ambig), MAX_VARIANTS), dtype=np.float32)

for model in models:
    model.eval()
    probs_fold = []
    with torch.no_grad():
        for batch in tqdm(test_ambig_loader, desc='Fold inference', leave=False):
            word, suffix, prefix, case, uidx = [x.to(device) for x in batch]
            logits, _ = model(word, suffix, prefix, case, uidx, char_matrix_tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            probs_fold.append(probs)
    probs_fold = np.vstack(probs_fold)
    all_probs += probs_fold

all_probs /= len(models)  # average

# Map to final after strings
ambig_predictions = []
for i, (_, row) in enumerate(test_ambig.iterrows()):
    before = row['before']
    cand = token_map.get(before, [])
    if not cand:  # fallback (should not happen for ambiguous tokens)
        pred_after = before
    else:
        probs_i = all_probs[i, :len(cand)]
        pred_idx = np.argmax(probs_i)
        pred_after = cand[pred_idx]
    ambig_predictions.append(pred_after)

# Assign to test_ambig
test_ambig['after'] = ambig_predictions

# For non‑ambiguous tokens, use the most frequent after (or the token itself if unknown)
default_after_map = {b: lst[0] for b, lst in token_map.items()}
test_nonambig['after'] = test_nonambig['before'].map(default_after_map).fillna(test_nonambig['before'])

# Combine back
test_full = pd.concat([test_ambig, test_nonambig], axis=0).sort_index()

# Create submission
test_full['id'] = test_full['sentence_id'].astype(str) + '_' + test_full['token_id'].astype(str)
submission = test_full[['id', 'after']]

# Ensure output directory
os.makedirs('./submission', exist_ok=True)
submission.to_csv('./submission/submission.csv', index=False, quoting=1)  # quote non‑numeric
print('Submission saved to ./submission/submission.csv')
print('All done.')