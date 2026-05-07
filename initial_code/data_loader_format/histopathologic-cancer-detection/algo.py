import os
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    """Set seed for DataLoader workers."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train model for one epoch."""
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
    """Validate model and return loss, AUC, and predictions."""
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
    """Train model with early stopping."""
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_auc = 0.0
    best_state = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
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
    """Generate predictions for validation set."""
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
    """Generate test predictions with Test Time Augmentation."""
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
            logits_r = model(torch.flip(images, dims=[1, 2]))
            p_r = torch.sigmoid(logits_r).flatten()
            # average
            batch_avg = (p + p_h + p_v + p_r) / 4.0
            probs.extend(batch_avg.cpu().numpy())
    return np.array(probs)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Histopathology Cancer Detection Training')
    
    # Model hyperparameters
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=8,
                        help='Number of epochs to train (default: 8)')
    parser.add_argument('--patience', type=int, default=3,
                        help='Early stopping patience (default: 3)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay for optimizer (default: 1e-5)')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44],
                        help='Seeds for model initialization ensemble (default: 42 43 44)')
    
    # Path parameters
    parser.add_argument('--input_dir', type=str, default='./input',
                        help='Input directory containing data (default: ./input)')
    parser.add_argument('--submission_path', type=str, default='./submission/submission.csv',
                        help='Path to save submission file (default: ./submission/submission.csv)')
    
    # Other parameters
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers (default: 4)')
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Set initial seed for reproducibility
    set_seed(42)
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directories
    os.makedirs(os.path.dirname(args.submission_path), exist_ok=True)
    os.makedirs("./working", exist_ok=True)
    
    # Initialize data loader and get data
    data_loader = MyDataLoader(input_dir=args.input_dir)
    train_data, test_data = data_loader.get_data()
    
    # Extract datasets and labels
    train_dataset = train_data['train_dataset']
    val_dataset = train_data['val_dataset']
    val_labels = train_data['val_labels']
    test_dataset = test_data['test_dataset']
    test_ids = test_data['test_ids']
    
    print(f"\nDataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Storage for predictions
    all_val_preds = []
    test_preds_sum = np.zeros(len(test_dataset))
    
    # Train each model with different seeds
    for i, seed in enumerate(args.seeds):
        set_seed(seed)
        print(f"\n{'='*50}")
        print(f"Training model {i+1} with seed {seed}")
        print(f"{'='*50}\n")
        
        # DataLoaders with reproducibility
        g = torch.Generator()
        g.manual_seed(seed)
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True,
            worker_init_fn=seed_worker, generator=g
        )
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True
        )
        
        # Model initialization
        model = models.efficientnet_b3(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
        model = model.to(device)
        
        # Train model
        best_auc = train_model(
            model, train_loader, val_loader, device,
            args.epochs, args.patience, args.lr, args.weight_decay,
            model_idx=i+1
        )
        print(f"Model {i+1} best validation AUC: {best_auc:.6f}")
        
        # Validation predictions for ensemble
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
    print("Ensemble Results:")
    for i, seed in enumerate(args.seeds):
        print(f"Model {i+1} (seed {seed}) AUC: {roc_auc_score(val_labels, all_val_preds[i]):.6f}")
    print(f"Ensemble Validation AUC: {ensemble_auc:.6f}")
    print("="*50)
    
    # Final test predictions (average over models)
    test_preds_final = test_preds_sum / len(args.seeds)
    
    # Create submission file
    submission = pd.DataFrame({'id': test_ids, 'label': test_preds_final})
    submission.to_csv(args.submission_path, index=False)
    print(f"\nSubmission saved to {args.submission_path}")


if __name__ == "__main__":
    main()