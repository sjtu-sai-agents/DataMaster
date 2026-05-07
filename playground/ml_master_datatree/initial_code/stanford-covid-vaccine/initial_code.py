import json
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42)

# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------
def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

# ----------------------------------------------------------------------
# Feature extraction and preprocessing
# ----------------------------------------------------------------------
SEQ_LEN = 107

def process_sample(sample, is_train=True):
    seq = sample['sequence']
    struct = sample['structure']
    loop = sample['predicted_loop_type']

    # Feature matrix (107, 16)
    feats = np.zeros((SEQ_LEN, 16), dtype=np.float32)

    # 1. Sequence one-hot (A,G,C,U) -> indices 0-3
    seq_map = {'A':0, 'G':1, 'C':2, 'U':3}
    for i, ch in enumerate(seq):
        if ch in seq_map:
            feats[i, seq_map[ch]] = 1.0

    # 2. Structure one-hot (.,(,)) -> indices 4-6
    struct_map = {'.':0, '(':1, ')':2}
    for i, ch in enumerate(struct):
        if ch in struct_map:
            feats[i, 4 + struct_map[ch]] = 1.0

    # 3. Predicted loop type one-hot (S,M,I,B,H,E,X) -> indices 7-13
    loop_map = {'S':0, 'M':1, 'I':2, 'B':3, 'H':4, 'E':5, 'X':6}
    for i, ch in enumerate(loop):
        if ch in loop_map:
            feats[i, 7 + loop_map[ch]] = 1.0   # 7 = 4+3

    # 4. Normalized position (i / SEQ_LEN) -> index 14
    feats[:, 14] = np.arange(SEQ_LEN) / SEQ_LEN

    # 5. Pairing distance -> index 15
    pairs = {}
    stack = []
    for i, ch in enumerate(struct):
        if ch == '(':
            stack.append(i)
        elif ch == ')':
            if stack:
                j = stack.pop()
                pairs[j] = i
                pairs[i] = j
    for i in range(SEQ_LEN):
        if i in pairs:
            feats[i, 15] = abs(i - pairs[i]) / SEQ_LEN
        else:
            feats[i, 15] = -1.0

    if not is_train:
        return feats, None, None, None

    # --- Targets and weights for training ---
    seq_scored = sample['seq_scored']   # always 68
    mask = np.zeros(SEQ_LEN, dtype=bool)
    mask[:seq_scored] = True

    # Targets (107,5)
    targets = np.zeros((SEQ_LEN, 5), dtype=np.float32)
    targets[:seq_scored, 0] = sample['reactivity']
    targets[:seq_scored, 1] = sample['deg_pH10']
    targets[:seq_scored, 2] = sample['deg_Mg_pH10']
    targets[:seq_scored, 3] = sample['deg_50C']
    targets[:seq_scored, 4] = sample['deg_Mg_50C']

    # Error weights
    errors = {
        0: sample['reactivity_error'],
        1: sample['deg_error_pH10'],
        2: sample['deg_error_Mg_pH10'],
        3: sample['deg_error_50C'],
        4: sample['deg_error_Mg_50C']
    }
    column_weight = np.array([1.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float32)   # only scored columns
    weights = np.zeros((SEQ_LEN, 5), dtype=np.float32)

    for col in range(5):
        err = np.array(errors[col], dtype=np.float32)
        err = np.clip(err, 0.01, 1.0)
        inv_err = 1.0 / err
        weights[:seq_scored, col] = inv_err * column_weight[col]

    return feats, targets, mask, weights


# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------
class RNADataset(Dataset):
    def __init__(self, features, targets=None, masks=None, weights=None, is_train=True):
        self.features = features
        self.is_train = is_train
        if is_train:
            self.targets = targets
            self.masks = masks
            self.weights = weights

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feats = torch.tensor(self.features[idx], dtype=torch.float32)
        if self.is_train:
            targs = torch.tensor(self.targets[idx], dtype=torch.float32)
            mask = torch.tensor(self.masks[idx], dtype=torch.bool)
            w = torch.tensor(self.weights[idx], dtype=torch.float32)
            return feats, targs, mask, w
        else:
            return feats


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------
class RNAModel(nn.Module):
    def __init__(self, input_dim=16, embed_dim=256, hidden_dim=256, num_layers=2, dropout=0.3):
        super().__init__()
        self.embed = nn.Linear(input_dim, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                            bidirectional=True, batch_first=True, dropout=dropout if num_layers>1 else 0)
        self.ln = nn.LayerNorm(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim * 2, 5)

    def forward(self, x):
        x = self.embed(x)
        x, _ = self.lstm(x)
        x = self.ln(x)
        x = self.dropout(x)
        x = self.out(x)
        return x


# ----------------------------------------------------------------------
# Evaluation metric (MCRMSE)
# ----------------------------------------------------------------------
def compute_mcrmse(model, dataloader, device, scored_cols=(0,2,4)):
    model.eval()
    sum_sq = {c: 0.0 for c in scored_cols}
    counts = {c: 0 for c in scored_cols}
    with torch.no_grad():
        for feats, targs, mask, _ in dataloader:
            feats = feats.to(device)
            targs = targs.to(device)
            mask = mask.to(device)
            preds = model(feats)
            for c in scored_cols:
                # Flatten and select positions with mask
                p = preds[:, :, c].reshape(-1)
                t = targs[:, :, c].reshape(-1)
                m = mask.reshape(-1)
                p_masked = p[m]
                t_masked = t[m]
                sum_sq[c] += ((p_masked - t_masked) ** 2).sum().item()
                counts[c] += m.sum().item()
    rmses = {}
    for c in scored_cols:
        if counts[c] > 0:
            rmses[c] = np.sqrt(sum_sq[c] / counts[c])
        else:
            rmses[c] = 0.0
    mcrmse = np.mean(list(rmses.values()))
    return mcrmse, rmses


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ---------- Load and preprocess training data ----------
    print("Loading training data...")
    train_raw = load_jsonl('./input/train.jsonl')
    all_feats, all_targs, all_masks, all_weights = [], [], [], []
    for sample in train_raw:
        f, t, m, w = process_sample(sample, is_train=True)
        all_feats.append(f)
        all_targs.append(t)
        all_masks.append(m)
        all_weights.append(w)

    # Split into train/validation
    indices = list(range(len(all_feats)))
    train_idx, val_idx = train_test_split(indices, test_size=0.1, random_state=42)
    full_dataset = RNADataset(all_feats, all_targs, all_masks, all_weights, is_train=True)
    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False,
                            num_workers=4, pin_memory=True)

    # ---------- Model, optimizer, scheduler ----------
    model = RNAModel().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

    best_val_mcrmse = float('inf')
    patience = 0
    max_patience = 5
    checkpoint_path = './working/best_model.pth'

    # ---------- Training loop ----------
    print("Start training...")
    for epoch in range(50):
        model.train()
        total_loss = 0.0
        for feats, targs, mask, weights in train_loader:
            feats = feats.to(device)
            targs = targs.to(device)
            weights = weights.to(device)

            optimizer.zero_grad()
            preds = model(feats)
            loss = torch.sum((preds - targs) ** 2 * weights) / (torch.sum(weights) + 1e-6)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        scheduler.step()

        val_mcrmse, _ = compute_mcrmse(model, val_loader, device)
        print(f"Epoch {epoch+1:2d} | Train loss: {avg_loss:.4f} | Val MCRMSE: {val_mcrmse:.6f}")

        if val_mcrmse < best_val_mcrmse:
            best_val_mcrmse = val_mcrmse
            torch.save(model.state_dict(), checkpoint_path)
            patience = 0
        else:
            patience += 1
            if patience >= max_patience:
                print(f"Early stopping after {epoch+1} epochs.")
                break

    # ---------- Load best model and final validation ----------
    model.load_state_dict(torch.load(checkpoint_path))
    final_mcrmse, per_col = compute_mcrmse(model, val_loader, device)
    print("\n" + "="*50)
    print(f"Validation MCRMSE: {final_mcrmse:.6f}")
    for col, rmse in per_col.items():
        name = ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C'][list(per_col.keys()).index(col)]
        print(f"  {name}: {rmse:.6f}")
    print("="*50)

    # ---------- Predict on test set and create submission ----------
    print("Loading test data...")
    test_raw = load_jsonl('./input/test.jsonl')
    test_ids = [s['id'] for s in test_raw]
    test_feats = [process_sample(s, is_train=False)[0] for s in test_raw]

    test_dataset = RNADataset(test_feats, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred = model(batch)      # (batch, 107, 5)
            all_preds.append(pred.cpu().numpy())
    all_preds = np.concatenate(all_preds, axis=0)   # (240, 107, 5)

    # Build submission dataframe
    rows = []
    for i, id_ in enumerate(test_ids):
        pred = all_preds[i]   # (107,5)
        for pos in range(SEQ_LEN):
            # columns: reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
            # we set deg_pH10 and deg_50C to 0.0 as per external knowledge
            rows.append([
                f"{id_}_{pos}",
                pred[pos, 0],               # reactivity
                pred[pos, 2],               # deg_Mg_pH10
                0.0,                        # deg_pH10
                pred[pos, 4],               # deg_Mg_50C
                0.0                         # deg_50C
            ])

    submission = pd.DataFrame(rows, columns=[
        'id_seqpos', 'reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C'
    ])

    os.makedirs('./submission', exist_ok=True)
    submission.to_csv('./submission/submission.csv', index=False)
    print("Submission saved to ./submission/submission.csv")


if __name__ == '__main__':
    main()