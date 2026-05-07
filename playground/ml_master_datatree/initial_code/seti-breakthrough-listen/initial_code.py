import os
import random
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import timm
from tqdm import tqdm

# -------------------- Configuration --------------------
class Config:
    seed = 42
    train_dir = "./input/train"
    test_dir = "./input/test"
    train_labels = "./input/train_labels.csv"
    sample_submission = "./input/sample_submission.csv"
    model_path = "./best_model.pth"
    submission_path = "./submission/submission.csv"
    img_size = 512
    batch_size = 32
    num_workers = 4
    lr = 1e-3
    epochs = 10
    mixup_alpha = 1.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------- Reproducibility --------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -------------------- Dataset --------------------
class SETIDataset(Dataset):
    def __init__(self, df, root_dir, is_test=False):
        self.df = df
        self.root_dir = root_dir
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        id_ = self.df.iloc[idx]['id']
        file_path = os.path.join(self.root_dir, id_[0], id_ + ".npy")
        try:
            data = np.load(file_path).astype(np.float32)   # (6, 273, 256)
        except FileNotFoundError:
            data = np.zeros((6, 273, 256), dtype=np.float32)

        # Stack the 6 cadence positions vertically
        data = np.vstack(data)                             # (1638, 256)

        # Resize to square image
        data = cv2.resize(data, (Config.img_size, Config.img_size),
                          interpolation=cv2.INTER_LINEAR)
        data = data.astype(np.float32)

        # Per‑sample standardization
        mean = data.mean()
        std = data.std()
        if std > 1e-6:
            data = (data - mean) / std
        else:
            data = data - mean

        # Add channel dimension
        data = data[np.newaxis, ...]                       # (1, H, W)

        image = torch.tensor(data, dtype=torch.float32)

        if self.is_test:
            return image
        else:
            target = self.df.iloc[idx]['target']
            target = torch.tensor(target, dtype=torch.float32)
            return image, target

# -------------------- Training & Validation --------------------
def train_one_epoch(model, loader, criterion, optimizer, scaler, mixup_alpha):
    model.train()
    running_loss = 0.0
    pbar = tqdm(loader, desc='Training', leave=False)
    for x, y in pbar:
        x = x.to(Config.device, non_blocking=True)
        y = y.to(Config.device, non_blocking=True).view(-1, 1)

        # Mixup augmentation
        if mixup_alpha > 0:
            lam = np.random.beta(mixup_alpha, mixup_alpha)
            index = torch.randperm(x.size(0)).to(Config.device)
            mixed_x = lam * x + (1 - lam) * x[index]
            y_a, y_b = y, y[index]
            with torch.cuda.amp.autocast():
                logits = model(mixed_x)
                loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
        else:
            with torch.cuda.amp.autocast():
                logits = model(x)
                loss = criterion(logits, y)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * x.size(0)
        pbar.set_postfix(loss=loss.item())

    return running_loss / len(loader.dataset)

def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_probs = []
    with torch.no_grad():
        for x, y in tqdm(loader, desc='Validation', leave=False):
            x = x.to(Config.device, non_blocking=True)
            y = y.to(Config.device, non_blocking=True).view(-1, 1)
            with torch.cuda.amp.autocast():
                logits = model(x)
                loss = criterion(logits, y)
            probs = torch.sigmoid(logits)
            running_loss += loss.item() * x.size(0)
            all_targets.append(y.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    loss_avg = running_loss / len(loader.dataset)
    all_targets = np.concatenate(all_targets, axis=0).squeeze()
    all_probs = np.concatenate(all_probs, axis=0).squeeze()
    auc = roc_auc_score(all_targets, all_probs) if len(np.unique(all_targets)) > 1 else 0.5
    return loss_avg, auc

# -------------------- Main --------------------
def main():
    set_seed(Config.seed)
    os.makedirs(os.path.dirname(Config.submission_path), exist_ok=True)

    # Load training labels
    train_df = pd.read_csv(Config.train_labels)
    print(f"Train samples: {len(train_df)}")

    # Stratified train/validation split (80/20)
    train_idx, val_idx = train_test_split(
        train_df.index, test_size=0.2,
        stratify=train_df['target'], random_state=Config.seed
    )
    df_train = train_df.iloc[train_idx].reset_index(drop=True)
    df_val = train_df.iloc[val_idx].reset_index(drop=True)
    print(f"Train size: {len(df_train)}, Validation size: {len(df_val)}")

    # Datasets and DataLoaders
    train_dataset = SETIDataset(df_train, Config.train_dir)
    val_dataset = SETIDataset(df_val, Config.train_dir)

    train_loader = DataLoader(
        train_dataset, batch_size=Config.batch_size, shuffle=True,
        num_workers=Config.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.batch_size, shuffle=False,
        num_workers=Config.num_workers, pin_memory=True
    )

    # Model
    model = timm.create_model('efficientnet_b0', pretrained=True,
                              in_chans=1, num_classes=1)
    model = model.to(Config.device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=Config.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.epochs)
    scaler = torch.cuda.amp.GradScaler()

    best_auc = 0.0

    # Training loop
    for epoch in range(1, Config.epochs + 1):
        print(f"\nEpoch {epoch}/{Config.epochs}")
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, Config.mixup_alpha
        )
        val_loss, val_auc = validate(model, val_loader, criterion)
        scheduler.step()

        print(f"Train Loss: {train_loss:.4f}  Val Loss: {val_loss:.4f}  Val AUC: {val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), Config.model_path)
            print(f"Saved best model with AUC {val_auc:.4f}")

    print(f"\nBest validation AUC: {best_auc:.4f}")

    # -------------------- Test Prediction --------------------
    print("Generating test predictions...")
    test_df = pd.read_csv(Config.sample_submission)
    test_dataset = SETIDataset(test_df, Config.test_dir, is_test=True)
    test_loader = DataLoader(
        test_dataset, batch_size=Config.batch_size, shuffle=False,
        num_workers=Config.num_workers, pin_memory=True
    )

    model.load_state_dict(torch.load(Config.model_path))
    model.eval()
    all_probs = []
    with torch.no_grad():
        for x in tqdm(test_loader, desc='Test'):
            x = x.to(Config.device, non_blocking=True)
            with torch.cuda.amp.autocast():
                logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)

    all_probs = np.concatenate(all_probs, axis=0).squeeze()
    test_df['target'] = all_probs
    test_df.to_csv(Config.submission_path, index=False)
    print(f"Submission saved to {Config.submission_path}")

if __name__ == "__main__":
    main()