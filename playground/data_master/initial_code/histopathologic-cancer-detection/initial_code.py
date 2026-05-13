import os
import random
import time
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# ------------------------------------------------------------
# Setup and reproducibility
# ------------------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# ------------------------------------------------------------
# Dataset classes
# ------------------------------------------------------------
class HistopathologyDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_id = self.dataframe.loc[idx, 'id']
        label = self.dataframe.loc[idx, 'label']
        img_path = os.path.join(self.image_dir, f"{img_id}.tif")
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(label, dtype=torch.float32)
        return image, label

class TestDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_id = self.dataframe.loc[idx, 'id']
        img_path = os.path.join(self.image_dir, f"{img_id}.tif")
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image, img_id

# ------------------------------------------------------------
# Training and evaluation functions
# ------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device).unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss

def validate(model, loader, criterion, device):
    model.eval()
    val_loss = 0.0
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validation", leave=False):
            images, labels = images.to(device), labels.to(device).unsqueeze(1)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            probs = torch.sigmoid(outputs).flatten().cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy().flatten())
    val_loss /= len(loader.dataset)
    auc = roc_auc_score(all_labels, all_probs)
    return val_loss, auc, np.array(all_probs)

def train_model(model, train_loader, val_loader, device, epochs, patience, lr, weight_decay, model_idx):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_auc = 0.0
    best_state = None
    no_improve = 0

    for epoch in range(1, epochs+1):
        print(f"Model {model_idx} Epoch {epoch}/{epochs}")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_auc, _ = validate(model, val_loader, criterion, device)
        scheduler.step()
        print(f"  train loss: {train_loss:.5f}  val loss: {val_loss:.5f}  val AUC: {val_auc:.5f}")

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = model.state_dict()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping after {epoch} epochs")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_auc

def predict_val(model, val_loader, device):
    model.eval()
    probs = []
    with torch.no_grad():
        for images, _ in val_loader:
            images = images.to(device)
            outputs = model(images)
            p = torch.sigmoid(outputs).flatten().cpu().numpy()
            probs.extend(p)
    return np.array(probs)

def predict_test_tta(model, test_loader, device):
    model.eval()
    probs = []
    with torch.no_grad():
        for images, _ in tqdm(test_loader, desc="Test TTA", leave=False):
            images = images.to(device)
            # original
            logits = model(images)
            p = torch.sigmoid(logits).flatten()
            # horizontal flip
            logits_h = model(torch.flip(images, dims=[2]))
            p_h = torch.sigmoid(logits_h).flatten()
            # vertical flip
            logits_v = model(torch.flip(images, dims=[1]))
            p_v = torch.sigmoid(logits_v).flatten()
            # 180° rotation (horizontal + vertical)
            logits_r = model(torch.flip(images, dims=[1,2]))
            p_r = torch.sigmoid(logits_r).flatten()
            # average
            batch_avg = (p + p_h + p_v + p_r) / 4.0
            probs.extend(batch_avg.cpu().numpy())
    return np.array(probs)

# ------------------------------------------------------------
# Main program
# ------------------------------------------------------------
def main():
    set_seed(42)   # for data splitting
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Paths
    input_dir = "./input"
    train_dir = os.path.join(input_dir, "train")
    test_dir = os.path.join(input_dir, "test")
    train_labels_path = os.path.join(input_dir, "train_labels.csv")
    sample_sub_path = os.path.join(input_dir, "sample_submission.csv")
    submission_path = "./submission/submission.csv"
    os.makedirs("./submission", exist_ok=True)
    os.makedirs("./working", exist_ok=True)

    # Load labels and split
    df = pd.read_csv(train_labels_path)
    train_df, val_df = train_test_split(df, test_size=0.1, stratify=df['label'], random_state=42)
    print(f"Train size: {len(train_df)}, Val size: {len(val_df)}")

    # Test IDs from sample submission (order must match)
    sample_sub = pd.read_csv(sample_sub_path)
    test_ids = sample_sub['id']
    test_df = pd.DataFrame({'id': test_ids})
    print(f"Test size: {len(test_df)}")

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    test_transform = val_transform   # base for TTA

    # Datasets (fixed splits)
    train_dataset = HistopathologyDataset(train_df, train_dir, transform=train_transform)
    val_dataset = HistopathologyDataset(val_df, train_dir, transform=val_transform)
    test_dataset = TestDataset(test_df, test_dir, transform=test_transform)

    # Validation labels (for final ensemble AUC)
    val_labels = val_df['label'].values

    # Hyperparameters
    batch_size = 64
    epochs = 8
    patience = 3
    lr = 0.001
    weight_decay = 1e-5
    seeds = [42, 43, 44]   # seeds for model initialisation

    # Storage for predictions
    all_val_preds = []
    test_preds_sum = np.zeros(len(test_df))

    # Train each model
    for i, seed in enumerate(seeds):
        set_seed(seed)
        print(f"\n===== Training model {i+1} with seed {seed} =====\n")

        # DataLoaders with reproducibility
        g = torch.Generator()
        g.manual_seed(seed)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=4, pin_memory=True,
            worker_init_fn=seed_worker, generator=g
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=True
        )

        # Model
        model = models.efficientnet_b3(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
        model = model.to(device)

        # Train
        best_auc = train_model(model, train_loader, val_loader, device,
                               epochs, patience, lr, weight_decay,
                               model_idx=i+1)
        print(f"Model {i+1} best validation AUC: {best_auc:.6f}")

        # Validation predictions (no TTA) for ensemble
        val_preds = predict_val(model, val_loader, device)
        all_val_preds.append(val_preds)

        # Test predictions with TTA
        test_preds = predict_test_tta(model, test_loader, device)
        test_preds_sum += test_preds

        # Cleanup
        del model
        torch.cuda.empty_cache()

    # Ensemble validation AUC
    val_preds_ensemble = np.mean(np.array(all_val_preds), axis=0)
    ensemble_auc = roc_auc_score(val_labels, val_preds_ensemble)
    print("\n" + "="*50)
    for i, seed in enumerate(seeds):
        print(f"Model {i+1} (seed {seed}) AUC: {roc_auc_score(val_labels, all_val_preds[i]):.6f}")
    print(f"Ensemble Validation AUC: {ensemble_auc:.6f}")
    print("="*50)

    # Final test predictions (average over models)
    test_preds_final = test_preds_sum / len(seeds)

    # Create submission file
    submission = pd.DataFrame({'id': test_ids, 'label': test_preds_final})
    submission.to_csv(submission_path, index=False)
    print(f"\nSubmission saved to {submission_path}")

if __name__ == "__main__":
    main()