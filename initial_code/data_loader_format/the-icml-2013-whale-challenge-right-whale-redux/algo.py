import os
import random
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import warnings
warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description='Audio Classification Training Script')
    
    # Path arguments
    parser.add_argument('--input_dir', type=str, default='./input',
                        help='Input directory containing train and test data')
    parser.add_argument('--submission_dir', type=str, default='./submission',
                        help='Directory to save submission files')
    parser.add_argument('--working_dir', type=str, default='./working',
                        help='Working directory for checkpoints')
    
    # Audio parameters
    parser.add_argument('--sample_rate', type=int, default=2000,
                        help='Audio sample rate')
    parser.add_argument('--duration', type=float, default=3.0,
                        help='Audio duration in seconds')
    parser.add_argument('--n_mels', type=int, default=64,
                        help='Number of mel bands')
    parser.add_argument('--n_fft', type=int, default=512,
                        help='FFT window size')
    parser.add_argument('--hop_length', type=int, default=128,
                        help='Hop length for STFT')
    parser.add_argument('--fmin', type=int, default=30,
                        help='Minimum frequency for mel filterbank')
    parser.add_argument('--fmax', type=int, default=800,
                        help='Maximum frequency for mel filterbank')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Number of data loading workers')
    parser.add_argument('--max_epochs', type=int, default=14,
                        help='Maximum number of training epochs')
    parser.add_argument('--early_stop_patience', type=int, default=5,
                        help='Early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for optimizer')
    parser.add_argument('--finetune_epochs', type=int, default=5,
                        help='Number of fine-tuning epochs')
    parser.add_argument('--finetune_lr', type=float, default=5e-4,
                        help='Learning rate for fine-tuning')
    
    # Model parameters
    parser.add_argument('--base_channels', type=int, default=32,
                        help='Base number of channels in model')
    parser.add_argument('--num_blocks', type=int, default=1,
                        help='Number of residual blocks per stage')
    
    # Other parameters
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--use_amp', action='store_true', default=True,
                        help='Use automatic mixed precision')
    parser.add_argument('--no_amp', action='store_false', dest='use_amp',
                        help='Disable automatic mixed precision')
    parser.add_argument('--tta_shifts', type=float, nargs='+', 
                        default=[0.0, 0.25, -0.25, 0.5, -0.5],
                        help='TTA time shifts in seconds')
    
    return parser.parse_args()


# ----------------------------------------------------------------------
# Model Definition
# ----------------------------------------------------------------------
class ResidualBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out


class Mel2DCNN(nn.Module):
    def __init__(self, input_channels=3, base_channels=32, num_blocks=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.relu = nn.ReLU(inplace=True)

        self.stage1 = self._make_layer(base_channels, base_channels, num_blocks, stride=1)
        self.stage2 = self._make_layer(base_channels, base_channels * 2, num_blocks, stride=2)
        self.stage3 = self._make_layer(base_channels * 2, base_channels * 4, num_blocks, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_channels * 4, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(ResidualBlock2D(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock2D(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# ----------------------------------------------------------------------
# Training Functions
# ----------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, criterion, scaler, device, use_amp):
    model.train()
    total_loss = 0.0
    for data, target in loader:
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True).view(-1, 1)
        optimizer.zero_grad()
        with autocast(enabled=use_amp):
            output = model(data)
            loss = criterion(output, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * data.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp):
    model.eval()
    total_loss = 0.0
    preds = []
    targets = []
    for data, target in loader:
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True).view(-1, 1)
        with autocast(enabled=use_amp):
            output = model(data)
            loss = criterion(output, target)
        total_loss += loss.item() * data.size(0)
        preds.append(torch.sigmoid(output).cpu().numpy())
        targets.append(target.cpu().numpy())
    preds = np.concatenate(preds, axis=0).squeeze()
    targets = np.concatenate(targets, axis=0).squeeze()
    auc = roc_auc_score(targets, preds) if len(np.unique(targets)) > 1 else 0.5
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, auc


def fit(model, train_loader, val_loader, optimizer, criterion, device, epochs, 
        patience, checkpoint_path, use_amp):
    best_auc = 0.0
    best_epoch = -1
    scaler = GradScaler(enabled=use_amp)

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device, use_amp)
        val_loss, val_auc = evaluate(model, val_loader, criterion, device, use_amp)
        print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f} - Val AUC: {val_auc:.6f}")

        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch
            torch.save(model.state_dict(), checkpoint_path)

        if epoch - best_epoch >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    model.load_state_dict(torch.load(checkpoint_path))
    return best_auc


def collate_tta(batch):
    """Custom collate for TTA dataset."""
    feats_shifts = []
    names = []
    for item in batch:
        feat_list, name = item
        feats_shifts.extend(feat_list)
        names.append(name)
    feats_batch = torch.stack(feats_shifts, dim=0)
    return feats_batch, names


def predict_test_tta(model, test_dataset, device, batch_size, num_workers, tta_shifts, use_amp):
    """Return dict {filename: probability} averaged over TTA shifts."""
    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True,
                        collate_fn=collate_tta)
    model.eval()
    all_probs = {}
    with torch.no_grad():
        for feats, names in loader:
            feats = feats.to(device, non_blocking=True)
            with autocast(enabled=use_amp):
                outputs = model(feats)
            probs = torch.sigmoid(outputs).cpu().numpy().squeeze()
            num_shifts = len(tta_shifts)
            for i, name in enumerate(names):
                start = i * num_shifts
                end = (i + 1) * num_shifts
                slice_probs = probs[start:end] if num_shifts > 1 else [probs[i]]
                avg_prob = np.mean(slice_probs)
                all_probs[name] = float(avg_prob)
    return all_probs


def main():
    args = parse_args()
    
    # Set random seeds
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create directories
    os.makedirs(args.submission_dir, exist_ok=True)
    os.makedirs(args.working_dir, exist_ok=True)

    # Number of workers
    num_workers = args.num_workers if args.num_workers else min(12, max(1, os.cpu_count() - 1))

    # Initialize DataLoader
    data_loader = MyDataLoader(
        input_dir=args.input_dir,
        sample_rate=args.sample_rate,
        duration=args.duration,
        n_mels=args.n_mels,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        fmin=args.fmin,
        fmax=args.fmax,
        random_seed=args.random_seed,
        batch_size=args.batch_size,
        num_workers=num_workers,
        tta_shifts=args.tta_shifts,
    )
    
    train_data, test_data = data_loader.get_data()
    
    train_loader = train_data['train_loader']
    val_loader = train_data['val_loader']
    train_labels = train_data['train_labels']
    val_paths = train_data['val_paths']
    val_labels = train_data['val_labels']
    tta_shifts = train_data['tta_shifts']
    
    test_loader = test_data['test_loader']
    test_paths = test_data['test_paths']

    # Compute pos_weight for BCEWithLogitsLoss
    pos_count = sum(train_labels)
    neg_count = len(train_labels) - pos_count
    pos_weight = torch.tensor([max(1.0, neg_count / pos_count)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Initialize model
    model = Mel2DCNN(input_channels=3, base_channels=args.base_channels, 
                     num_blocks=args.num_blocks).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # Checkpoint path
    ckpt_path = os.path.join(args.working_dir, "best_model.pth")

    # Train with early stopping
    print("Starting training...")
    best_val_auc = fit(model, train_loader, val_loader, optimizer, criterion, device,
                       args.max_epochs, args.early_stop_patience, ckpt_path, args.use_amp)
    print(f"Best validation AUC: {best_val_auc:.6f}")

    # Fine-tune on full training data
    print("Fine-tuning on full training set...")
    full_train_loader = data_loader.get_full_train_loader()
    optimizer_ft = optim.AdamW(model.parameters(), lr=args.finetune_lr, weight_decay=args.weight_decay)
    scaler_ft = GradScaler(enabled=args.use_amp)

    for epoch in range(args.finetune_epochs):
        train_loss = train_one_epoch(model, full_train_loader, optimizer_ft, criterion, 
                                      scaler_ft, device, args.use_amp)
        print(f"Finetune epoch {epoch + 1}/{args.finetune_epochs} - Train Loss: {train_loss:.6f}")

    # Evaluate final model on validation set
    _, val_auc = evaluate(model, val_loader, criterion, device, args.use_amp)
    print(f"Validation AUC (no TTA): {val_auc:.6f}")

    # Validation with TTA
    val_tta_dataset = data_loader.get_val_tta_dataset()
    val_tta_probs = predict_test_tta(model, val_tta_dataset, device, args.batch_size, 
                                      num_workers, tta_shifts, args.use_amp)
    val_tta_preds = np.array([val_tta_probs[os.path.basename(p)] for p in val_paths])
    val_auc_tta = roc_auc_score(val_labels, val_tta_preds)
    print(f"Validation AUC (with TTA): {val_auc_tta:.6f}")

    # Test set predictions with TTA
    test_probs = predict_test_tta(model, test_loader.dataset, device, args.batch_size,
                                   num_workers, tta_shifts, args.use_amp)

    # Build submission
    submission = pd.DataFrame({
        "clip": [os.path.basename(p) for p in test_paths],
        "probability": [test_probs[os.path.basename(p)] for p in test_paths]
    })
    submission = submission.sort_values("clip")
    submission.to_csv(os.path.join(args.submission_dir, "submission.csv"), index=False)
    print(f"Submission saved to {args.submission_dir}/submission.csv")

    print("Script finished.")


if __name__ == "__main__":
    main()