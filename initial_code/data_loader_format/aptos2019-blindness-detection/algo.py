import os
import random
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import timm


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Diabetic Retinopathy Detection Training')
    
    # Model hyperparameters
    parser.add_argument('--model_name', type=str, default='tf_efficientnet_b5_ns',
                        help='Model architecture name from timm')
    parser.add_argument('--img_size', type=int, default=456,
                        help='Input image size')
    parser.add_argument('--num_classes', type=int, default=4,
                        help='Number of output classes for ordinal regression')
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=7,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--wd', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--mixup_alpha', type=float, default=0.2,
                        help='Mixup alpha parameter (0 to disable)')
    
    # System parameters
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Number of data loading workers (default: min(8, cpu_count))')
    
    # Paths
    parser.add_argument('--submission_dir', type=str, default='./submission',
                        help='Submission output directory')
    parser.add_argument('--working_dir', type=str, default='./working',
                        help='Working directory for checkpoints')
    
    # Other
    parser.add_argument('--tta', action='store_true', default=True,
                        help='Use test time augmentation (horizontal flip)')
    
    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def mixup_data(x, y, alpha, device):
    """Apply mixup augmentation to batch."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def ordinal_predict(logits):
    """Convert ordinal logits to class predictions."""
    probs = torch.sigmoid(logits)
    pred_class = (probs > 0.5).sum(dim=1)
    return pred_class


def compute_qwk(true, pred):
    """Compute quadratic weighted kappa."""
    return cohen_kappa_score(true, pred, weights='quadratic')


def apply_thresholds(continuous, thresholds):
    """Apply thresholds to continuous predictions for class assignment."""
    pred_class = np.zeros_like(continuous, dtype=int)
    for i, s in enumerate(continuous):
        pred_class[i] = sum(s > t for t in thresholds)
    return pred_class


def predict_test(model, loader, device, use_tta=True):
    """Generate predictions on test set with optional TTA."""
    model.eval()
    all_ids = []
    all_continuous = []
    
    with torch.no_grad():
        for images, _, ids in loader:
            images = images.to(device, non_blocking=True)
            
            # Original prediction
            with torch.cuda.amp.autocast():
                logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            continuous = probs.sum(axis=1)
            
            if use_tta:
                # Horizontal flip TTA
                flipped_images = torch.flip(images, dims=[3])
                with torch.cuda.amp.autocast():
                    logits_f = model(flipped_images)
                probs_f = torch.sigmoid(logits_f).cpu().numpy()
                continuous_f = probs_f.sum(axis=1)
                continuous = (continuous + continuous_f) / 2.0
            
            all_continuous.extend(continuous)
            all_ids.extend(ids)
    
    return all_ids, np.array(all_continuous)


def main():
    """Main training function."""
    args = parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs(args.submission_dir, exist_ok=True)
    os.makedirs(args.working_dir, exist_ok=True)
    
    # Initialize data loader
    num_workers = args.num_workers if args.num_workers else min(8, os.cpu_count())
    data_loader = MyDataLoader(
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=num_workers,
        seed=args.seed
    )
    
    # Get data
    train_data, test_loader = data_loader.get_data()
    train_loader, val_loader, pos_weight = train_data
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Positive weights: {pos_weight}")
    
    # Initialize model
    model = timm.create_model(args.model_name, pretrained=True, num_classes=args.num_classes)
    model = model.to(device)
    
    # Loss, optimizer, scheduler
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler()
    
    # Training loop
    best_qwk = -1.0
    checkpoint_file = os.path.join(args.working_dir, 'best_model.pth')
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        
        for images, targets, _ in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            if args.mixup_alpha > 0:
                mixed_images, targets_a, targets_b, lam = mixup_data(
                    images, targets, args.mixup_alpha, device
                )
                with torch.cuda.amp.autocast():
                    logits = model(mixed_images)
                    loss = lam * criterion(logits, targets_a) + (1 - lam) * criterion(logits, targets_b)
            else:
                with torch.cuda.amp.autocast():
                    logits = model(images)
                    loss = criterion(logits, targets)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item() * images.size(0)
        
        scheduler.step()
        train_loss = total_loss / len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_preds = []
        val_true = []
        
        with torch.no_grad():
            for images, targets, _ in val_loader:
                images = images.to(device, non_blocking=True)
                with torch.cuda.amp.autocast():
                    logits = model(images)
                pred_class = ordinal_predict(logits).cpu().numpy()
                val_preds.extend(pred_class)
                true_class = targets.sum(dim=1).cpu().numpy()
                val_true.extend(true_class)
        
        qwk = compute_qwk(val_true, val_preds)
        print(f'Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f} | Val QWK: {qwk:.4f}')
        
        if qwk > best_qwk:
            best_qwk = qwk
            torch.save(model.state_dict(), checkpoint_file)
            print(f'*** New best model saved (QWK = {qwk:.4f})')
    
    # Load best model
    model.load_state_dict(torch.load(checkpoint_file))
    model.eval()
    
    # Threshold optimization
    print("\nOptimizing thresholds...")
    val_continuous = []
    val_true_labels = []
    
    with torch.no_grad():
        for images, targets, _ in val_loader:
            images = images.to(device, non_blocking=True)
            with torch.cuda.amp.autocast():
                logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            cont = probs.sum(axis=1)
            val_continuous.extend(cont)
            true_labels = targets.sum(dim=1).cpu().numpy()
            val_true_labels.extend(true_labels)
    
    val_continuous = np.array(val_continuous)
    val_true_labels = np.array(val_true_labels, dtype=int)
    
    # Grid search for optimal thresholds
    initial_thresholds = np.array([0.5, 1.5, 2.5, 3.5])
    best_thresholds = initial_thresholds.copy()
    best_qwk_opt = compute_qwk(val_true_labels, apply_thresholds(val_continuous, initial_thresholds))
    
    span = 0.75
    points = 41
    
    for iteration in range(3):
        print(f'Threshold optimization iteration {iteration+1}')
        for i in range(4):
            low = best_thresholds[i] - span
            high = best_thresholds[i] + span
            if i > 0:
                low = max(low, best_thresholds[i-1])
            if i < 3:
                high = min(high, best_thresholds[i+1])
            if low >= high:
                low = high = best_thresholds[i]
            
            grid = np.linspace(low, high, points)
            local_best_qwk = best_qwk_opt
            local_best_t = best_thresholds[i]
            
            for t in grid:
                test_thresholds = best_thresholds.copy()
                test_thresholds[i] = t
                pred_class = apply_thresholds(val_continuous, test_thresholds)
                qwk = compute_qwk(val_true_labels, pred_class)
                if qwk > local_best_qwk:
                    local_best_qwk = qwk
                    local_best_t = t
            
            best_thresholds[i] = local_best_t
            best_qwk_opt = max(best_qwk_opt, local_best_qwk)
    
    print(f'Optimized thresholds: {best_thresholds}')
    print(f'Validation QWK with optimized thresholds: {best_qwk_opt:.4f}')
    
    # Test inference
    print("\nGenerating test predictions...")
    test_ids, test_continuous = predict_test(model, test_loader, device, use_tta=args.tta)
    test_preds = apply_thresholds(test_continuous, best_thresholds)
    
    # Save submission
    submission_file = os.path.join(args.submission_dir, 'submission.csv')
    submission = pd.DataFrame({'id_code': test_ids, 'diagnosis': test_preds})
    submission.to_csv(submission_file, index=False)
    print(f'Submission saved to {submission_file}')
    
    # Print final metrics
    print(f'\nFinal Results:')
    print(f'Validation QWK (simple rule): {best_qwk:.4f}')
    print(f'Validation QWK (optimized thresholds): {best_qwk_opt:.4f}')


if __name__ == "__main__":
    main()