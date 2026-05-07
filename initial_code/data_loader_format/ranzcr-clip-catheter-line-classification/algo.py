import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import timm
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


class EfficientNetModel(nn.Module):
    """EfficientNet-based model for multi-label classification."""
    
    def __init__(self, num_classes=11, model_name="efficientnet_b3", dropout=0.3):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        in_features = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


def train_epoch(model, loader, criterion, optimizer, scaler, device, use_amp=True):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    
    for images, labels in tqdm(loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        with autocast(enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
    
    return running_loss / len(loader.dataset)


def validate(model, loader, criterion, device, target_cols):
    """Validate the model and compute AUC scores."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            all_preds.append(torch.sigmoid(outputs).cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # Compute AUC per class
    auc_scores = []
    for i in range(all_labels.shape[1]):
        try:
            auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
            auc_scores.append(auc)
        except ValueError:
            auc_scores.append(0.5)

    avg_auc = np.mean(auc_scores)
    return running_loss / len(loader.dataset), avg_auc, auc_scores


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train EfficientNet for Chest X-Ray Classification")
    
    # Path arguments
    parser.add_argument('--data_dir', type=str, default='./input',
                        help='Data directory containing train.csv, test/, etc.')
    parser.add_argument('--working_dir', type=str, default='./working',
                        help='Working directory for saving models')
    parser.add_argument('--submission_path', type=str, default='./submission/submission.csv',
                        help='Path to save submission file')
    
    # Model hyperparameters
    parser.add_argument('--img_size', type=int, default=384,
                        help='Image size for resizing')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=8,
                        help='Number of training epochs')
    parser.add_argument('--folds', type=int, default=5,
                        help='Number of cross-validation folds (used when val.csv not available)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay for optimizer')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate in classifier')
    parser.add_argument('--model_name', type=str, default='efficientnet_b3',
                        help='Model backbone name (timm format)')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--use_amp', action='store_true', default=True,
                        help='Use automatic mixed precision')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loader workers')
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Create directories
    os.makedirs(args.working_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.submission_path), exist_ok=True)
    
    # Load data using MyDataLoader
    data_loader = MyDataLoader(
        data_dir=args.data_dir,
        img_size=args.img_size,
        folds=args.folds,
        seed=args.seed
    )
    train_data, test_data = data_loader.get_data()
    
    # Extract training data
    train_df = train_data['df']
    val_df = train_data['val_df']
    train_transform = train_data['train_transform']
    val_transform = train_data['val_transform']
    train_dir = train_data['img_dir']
    target_cols = train_data['target_cols']
    folds = train_data['folds']
    has_val_csv = train_data['has_val_csv']
    
    # Extract test data
    test_df = test_data['df']
    test_transform = test_data['transform']
    test_dir = test_data['img_dir']
    
    print(f"\nTarget columns: {len(target_cols)}")
    print(f"Training samples: {len(train_df)}")
    print(f"Using fixed val.csv: {has_val_csv}")
    print(f"Number of folds: {folds}")
    
    # Training
    fold_aucs = []
    use_amp = args.use_amp and torch.cuda.is_available()
    
    if has_val_csv:
        # Train on all training data with fixed validation set
        print("\n" + "="*50)
        print("Training with fixed validation set from val.csv")
        print("="*50)
        
        # Compute class weights for imbalanced data
        pos_counts = train_df[target_cols].sum()
        total = len(train_df)
        class_weights = (total - pos_counts) / (pos_counts + 1e-7)
        class_weights = torch.tensor(class_weights.values, dtype=torch.float32).to(device)
        
        # Create datasets and dataloaders
        train_dataset = ChestXRayDataset(train_df, train_dir, transform=train_transform, target_cols=target_cols)
        val_dataset = ChestXRayDataset(val_df, train_dir, transform=val_transform, target_cols=target_cols)
        
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size * 2, shuffle=False,
            num_workers=args.num_workers, pin_memory=True
        )
        
        # Initialize model, loss, optimizer, scheduler
        model = EfficientNetModel(
            num_classes=len(target_cols),
            model_name=args.model_name,
            dropout=args.dropout
        ).to(device)
        
        criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        scaler = GradScaler() if use_amp else None
        
        best_auc = 0.0
        for epoch in range(args.epochs):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, device, use_amp)
            val_loss, val_auc, _ = validate(model, val_loader, criterion, device, target_cols)
            scheduler.step()
            
            print(f"Epoch {epoch+1}/{args.epochs}: Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")
            
            if val_auc > best_auc:
                best_auc = val_auc
                torch.save(model.state_dict(), os.path.join(args.working_dir, "best_model.pth"))
        
        fold_aucs.append(best_auc)
        print(f"\nBest Validation AUC: {best_auc:.4f}")
        
        model_paths = [os.path.join(args.working_dir, "best_model.pth")]
        
    else:
        # Cross-validation training
        print("\n" + "="*50)
        print("Training with GroupKFold cross-validation")
        print("="*50)
        
        for fold in range(folds):
            print(f"\n{'='*20} Fold {fold} {'='*20}")
            
            # Split data by fold
            train_fold = train_df[train_df["fold"] != fold]
            val_fold = train_df[train_df["fold"] == fold]
            
            # Compute class weights
            pos_counts = train_fold[target_cols].sum()
            total = len(train_fold)
            class_weights = (total - pos_counts) / (pos_counts + 1e-7)
            class_weights = torch.tensor(class_weights.values, dtype=torch.float32).to(device)
            
            # Create datasets and dataloaders
            train_dataset = ChestXRayDataset(train_fold, train_dir, transform=train_transform, target_cols=target_cols)
            val_dataset = ChestXRayDataset(val_fold, train_dir, transform=val_transform, target_cols=target_cols)
            
            train_loader = DataLoader(
                train_dataset, batch_size=args.batch_size, shuffle=True,
                num_workers=args.num_workers, pin_memory=True
            )
            val_loader = DataLoader(
                val_dataset, batch_size=args.batch_size * 2, shuffle=False,
                num_workers=args.num_workers, pin_memory=True
            )
            
            # Initialize model, loss, optimizer, scheduler
            model = EfficientNetModel(
                num_classes=len(target_cols),
                model_name=args.model_name,
                dropout=args.dropout
            ).to(device)
            
            criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
            optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
            scaler = GradScaler() if use_amp else None
            
            best_auc = 0.0
            for epoch in range(args.epochs):
                train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, device, use_amp)
                val_loss, val_auc, _ = validate(model, val_loader, criterion, device, target_cols)
                scheduler.step()
                
                print(f"Epoch {epoch+1}/{args.epochs}: Train Loss: {train_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")
                
                if val_auc > best_auc:
                    best_auc = val_auc
                    torch.save(model.state_dict(), os.path.join(args.working_dir, f"best_fold{fold}.pth"))
            
            fold_aucs.append(best_auc)
            print(f"Fold {fold} Best AUC: {best_auc:.4f}")
        
        print(f"\nCross-validation AUCs: {fold_aucs}")
        print(f"Mean CV AUC: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
        
        model_paths = [os.path.join(args.working_dir, f"best_fold{fold}.pth") for fold in range(folds)]
    
    # Generate ensemble predictions on test set
    print("\n" + "="*50)
    print("Generating ensemble predictions on test set")
    print("="*50)
    
    test_dataset = ChestXRayDataset(test_df, test_dir, transform=test_transform, target_cols=target_cols)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.num_workers
    )
    
    all_fold_preds = []
    for i, model_path in enumerate(model_paths):
        print(f"Loading model {i+1}/{len(model_paths)}: {model_path}")
        
        model = EfficientNetModel(
            num_classes=len(target_cols),
            model_name=args.model_name,
            dropout=args.dropout
        ).to(device)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        fold_preds = []
        with torch.no_grad():
            for images, _ in tqdm(test_loader, desc=f"Inference model {i+1}"):
                images = images.to(device)
                outputs = model(images)
                preds = torch.sigmoid(outputs).cpu().numpy()
                fold_preds.append(preds)
        
        fold_preds = np.concatenate(fold_preds)
        all_fold_preds.append(fold_preds)
    
    # Average predictions from all models
    ensemble_preds = np.mean(all_fold_preds, axis=0)
    
    # Create submission file
    submission = pd.DataFrame(ensemble_preds, columns=target_cols)
    submission.insert(0, "StudyInstanceUID", test_df["StudyInstanceUID"].values)
    
    # Ensure correct column order
    required_columns = ["StudyInstanceUID"] + target_cols
    submission = submission[required_columns]
    
    submission.to_csv(args.submission_path, index=False)
    print(f"\nSubmission saved to {args.submission_path}")
    print(f"Submission shape: {submission.shape}")
    print("\nFirst few rows of submission:")
    print(submission.head())
    
    print("\n" + "="*50)
    print("Training completed successfully!")
    print("="*50)


if __name__ == "__main__":
    main()