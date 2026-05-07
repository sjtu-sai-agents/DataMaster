import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class DRDataset(Dataset):
    """Diabetic Retinopathy Dataset for ordinal regression."""
    
    def __init__(self, df, img_dir, transform=None, has_labels=True):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.has_labels = has_labels

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['id_code'] + '.png')
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        if self.has_labels:
            label = row['diagnosis']
            # Ordinal targets: 4 binary tasks
            targets = torch.zeros(4, dtype=torch.float32)
            for k in range(4):
                targets[k] = 1.0 if label > k else 0.0
            return image, targets, row['id_code']
        else:
            return image, -1, row['id_code']


class MyDataLoader(BaseDataLoader):
    """DataLoader for Diabetic Retinopathy detection task."""
    
    def __init__(self, img_size=456, batch_size=8, num_workers=None, seed=42, **kwargs):
        super().__init__(**kwargs)
        self.img_size = img_size
        self.batch_size = batch_size
        self.seed = seed
        self.num_workers = num_workers if num_workers is not None else min(8, os.cpu_count())
        
        # Paths
        self.input_dir = "./input"
        self.train_img_dir = os.path.join(self.input_dir, "train_images")
        self.test_img_dir = os.path.join(self.input_dir, "test_images")
        self.train_csv = os.path.join(self.input_dir, "train.csv")
        self.test_csv = os.path.join(self.input_dir, "test.csv")
        self.val_csv = os.path.join(self.input_dir, "val.csv")
        
        # Define transforms
        self._setup_transforms()

    def _setup_transforms(self):
        """Setup data augmentation transforms."""
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        
        self.train_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    def _get_pos_weights(self, df, device):
        """Compute positive weights for ordinal regression loss."""
        labels = df['diagnosis'].values
        pos_counts = []
        neg_counts = []
        for k in range(4):
            pos = (labels > k).sum()
            neg = len(labels) - pos
            pos_counts.append(pos)
            neg_counts.append(neg)
        return torch.tensor([neg/pos for pos, neg in zip(pos_counts, neg_counts)], device=device)

    def setup(self):
        """Load data, perform train/val split, and create data loaders."""
        # Load data
        train_df = pd.read_csv(self.train_csv)
        test_df = pd.read_csv(self.test_csv)
        
        # Check for validation set - use fixed val.csv if exists
        if os.path.exists(self.val_csv):
            val_df = pd.read_csv(self.val_csv)
            # Handle different column names for image ID
            id_col = 'id_code' if 'id_code' in val_df.columns else 'image'
            val_images = set(val_df[id_col].values)
            train_df_split = train_df[~train_df['id_code'].isin(val_images)].reset_index(drop=True)
            # Ensure val_df has 'id_code' column for DRDataset
            if 'image' in val_df.columns and 'id_code' not in val_df.columns:
                val_df = val_df.rename(columns={'image': 'id_code'})
            val_df_split = val_df.reset_index(drop=True)
        else:
            # Use stratified split if no val.csv
            from sklearn.model_selection import StratifiedShuffleSplit
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=self.seed)
            for train_idx, val_idx in splitter.split(train_df['id_code'], train_df['diagnosis']):
                train_df_split = train_df.iloc[train_idx].reset_index(drop=True)
                val_df_split = train_df.iloc[val_idx].reset_index(drop=True)
        
        # Compute positive weights
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pos_weight = self._get_pos_weights(train_df_split, device)
        
        # Create datasets
        train_dataset = DRDataset(train_df_split, self.train_img_dir, 
                                  transform=self.train_transform, has_labels=True)
        val_dataset = DRDataset(val_df_split, self.train_img_dir, 
                                transform=self.val_transform, has_labels=True)
        test_dataset = DRDataset(test_df, self.test_img_dir, 
                                 transform=self.val_transform, has_labels=False)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True, 
            persistent_workers=self.num_workers > 0, drop_last=False
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
            persistent_workers=self.num_workers > 0
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
            persistent_workers=self.num_workers > 0
        )
        
        # Store data: train_data contains (train_loader, val_loader, pos_weight)
        self.train_data = (train_loader, val_loader, pos_weight)
        self.test_data = test_loader

    def describe(self):
        """Return description of the data processing approach."""
        return ("DataLoader for Diabetic Retinopathy detection. "
                "Uses EfficientNet-B5 compatible image size (456x456). "
                "Includes data augmentation (horizontal flip, vertical flip, color jitter). "
                "Implements ordinal regression targets (4 binary classifiers). "
                "Uses fixed validation set from val.csv if available, otherwise stratified split. "
                "Computes positive weights for balanced ordinal loss.")

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