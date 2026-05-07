import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm
import argparse
import warnings

warnings.filterwarnings("ignore")


class TabularDataset(Dataset):
    def __init__(self, df, cat_cols, cont_cols, target=None):
        self.cat_data = df[cat_cols].values.astype(np.int64)
        self.cont_data = df[cont_cols].values.astype(np.float32)
        self.target = target.astype(np.float32) if target is not None else None

    def __len__(self):
        return len(self.cat_data)

    def __getitem__(self, idx):
        if self.target is not None:
            return (
                torch.tensor(self.cat_data[idx], dtype=torch.long),
                torch.tensor(self.cont_data[idx], dtype=torch.float32),
                torch.tensor(self.target[idx], dtype=torch.float32),
            )
        else:
            return (
                torch.tensor(self.cat_data[idx], dtype=torch.long),
                torch.tensor(self.cont_data[idx], dtype=torch.float32),
            )


class TabularAttentionNN(nn.Module):
    def __init__(self, cat_dims, n_cont, hidden_dim=256, dropout=0.3):
        super().__init__()

        # Embedding layers for categorical features
        self.embeddings = nn.ModuleList(
            [nn.Embedding(dim, min(50, (dim + 1) // 2)) for dim in cat_dims]
        )

        emb_dim = sum([min(50, (dim + 1) // 2) for dim in cat_dims])
        total_dim = emb_dim + n_cont

        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            total_dim, num_heads=8, batch_first=True, dropout=dropout
        )

        # Main network
        self.network = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(self, cat_data, cont_data):
        # Process categorical features through embeddings
        cat_embedded = [emb(cat_data[:, i]) for i, emb in enumerate(self.embeddings)]
        cat_embedded = torch.cat(cat_embedded, dim=1)

        # Combine with continuous features
        x = torch.cat([cat_embedded, cont_data], dim=1)

        # Apply attention (add sequence dimension)
        x_attn = x.unsqueeze(1)
        x_attn, _ = self.attention(x_attn, x_attn, x_attn)
        x = x + x_attn.squeeze(1)  # Residual connection

        # Pass through main network
        return torch.sigmoid(self.network(x)).squeeze()


def parse_args():
    parser = argparse.ArgumentParser(description='Tabular Neural Network Training')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for training')
    parser.add_argument('--val_batch_size', type=int, default=2048, help='Batch size for validation')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs per fold')
    parser.add_argument('--final_epochs', type=int, default=20, help='Number of epochs for final model')
    parser.add_argument('--n_folds', type=int, default=5, help='Number of folds for cross-validation')
    parser.add_argument('--max_patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for DataLoader')
    parser.add_argument('--input_dir', type=str, default='./input', help='Input directory')
    parser.add_argument('--output_dir', type=str, default='./submission', help='Output directory')
    parser.add_argument('--working_dir', type=str, default='./working', help='Working directory for model checkpoints')
    return parser.parse_args()


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    train_loss = 0
    train_preds = []
    train_targets = []
    
    for cat_batch, cont_batch, target_batch in train_loader:
        cat_batch, cont_batch, target_batch = (
            cat_batch.to(device),
            cont_batch.to(device),
            target_batch.to(device),
        )
        
        optimizer.zero_grad()
        preds = model(cat_batch, cont_batch)
        loss = criterion(preds, target_batch)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        train_loss += loss.item()
        train_preds.extend(preds.detach().cpu().numpy())
        train_targets.extend(target_batch.cpu().numpy())
    
    return train_loss, train_preds, train_targets


def validate(model, val_loader, device):
    """Validate the model"""
    model.eval()
    val_preds = []
    val_targets = []
    
    with torch.no_grad():
        for cat_batch, cont_batch, target_batch in val_loader:
            cat_batch, cont_batch, target_batch = (
                cat_batch.to(device),
                cont_batch.to(device),
                target_batch.to(device),
            )
            preds = model(cat_batch, cont_batch)
            val_preds.extend(preds.cpu().numpy())
            val_targets.extend(target_batch.cpu().numpy())
    
    return val_preds, val_targets


def predict(model, dataset, batch_size, num_workers, device):
    """Make predictions on a dataset"""
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for cat_batch, cont_batch in DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        ):
            cat_batch, cont_batch = cat_batch.to(device), cont_batch.to(device)
            preds = model(cat_batch, cont_batch)
            predictions.extend(preds.cpu().numpy())
    
    return np.array(predictions)


def main():
    args = parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Get data from DataLoader
    data_loader = MyDataLoader()
    train_data, test_data = data_loader.get_data()
    
    # Extract data
    X = train_data['X']
    y = train_data['y']
    X_val = train_data['X_val']
    y_val = train_data['y_val']
    has_val = train_data['has_val']
    cat_cols = train_data['cat_cols']
    cont_cols = train_data['cont_cols']
    cat_dims = train_data['cat_dims']
    n_cont = train_data['n_cont']
    
    X_test = test_data['X_test']
    test_ids = test_data['test_ids']
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create working directory
    os.makedirs(args.working_dir, exist_ok=True)
    
    if has_val:
        # Use fixed validation set from val.csv
        print("Using fixed validation set from val.csv...")
        print(f"Training samples: {len(X)}, Validation samples: {len(X_val)}")
        
        # Create datasets and dataloaders
        train_dataset = TabularDataset(X, cat_cols, cont_cols, y)
        val_dataset = TabularDataset(X_val, cat_cols, cont_cols, y_val)
        
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, 
            num_workers=args.num_workers, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=args.val_batch_size, shuffle=False, 
            num_workers=args.num_workers, pin_memory=True
        )
        
        # Initialize model
        model = TabularAttentionNN(cat_dims, n_cont, args.hidden_dim, args.dropout).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=3, factor=0.5
        )
        criterion = nn.BCELoss()
        
        # Training loop
        best_val_auc = 0
        patience_counter = 0
        
        for epoch in range(args.epochs):
            train_loss, train_preds, train_targets = train_epoch(
                model, train_loader, optimizer, criterion, device
            )
            val_preds, val_targets = validate(model, val_loader, device)
            
            train_auc = roc_auc_score(train_targets, train_preds)
            val_auc = roc_auc_score(val_targets, val_preds)
            
            scheduler.step(val_auc)
            print(f"Epoch {epoch+1}: Train AUC = {train_auc:.6f}, Val AUC = {val_auc:.6f}")
            
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
                torch.save(model.state_dict(), os.path.join(args.working_dir, "model_best.pt"))
            else:
                patience_counter += 1
            
            if patience_counter >= args.max_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        model.load_state_dict(torch.load(os.path.join(args.working_dir, "model_best.pt")))
        
        # Make test predictions
        test_dataset = TabularDataset(X_test, cat_cols, cont_cols)
        test_preds = predict(model, test_dataset, args.val_batch_size, args.num_workers, device)
        
        overall_auc = best_val_auc
        print(f"\n{'='*50}")
        print(f"Validation AUC: {overall_auc:.6f}")
        
    else:
        # Use cross-validation
        print(f"Starting {args.n_folds}-fold cross-validation with neural network...")
        skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
        
        fold_scores = []
        test_preds = np.zeros(len(X_test))
        oof_preds = np.zeros(len(X))
        models = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            print(f"\nFold {fold}/{args.n_folds}")
            
            # Split data
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            # Create datasets and dataloaders
            train_dataset = TabularDataset(X_train_fold, cat_cols, cont_cols, y_train_fold)
            val_dataset = TabularDataset(X_val_fold, cat_cols, cont_cols, y_val_fold)
            
            train_loader = DataLoader(
                train_dataset, batch_size=args.batch_size, shuffle=True, 
                num_workers=args.num_workers, pin_memory=True
            )
            val_loader = DataLoader(
                val_dataset, batch_size=args.val_batch_size, shuffle=False, 
                num_workers=args.num_workers, pin_memory=True
            )
            
            # Initialize model
            model = TabularAttentionNN(cat_dims, n_cont, args.hidden_dim, args.dropout).to(device)
            optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", patience=3, factor=0.5
            )
            criterion = nn.BCELoss()
            
            # Training loop
            best_val_auc = 0
            patience_counter = 0
            
            for epoch in range(args.epochs):
                train_loss, train_preds, train_targets = train_epoch(
                    model, train_loader, optimizer, criterion, device
                )
                val_preds, val_targets = validate(model, val_loader, device)
                
                train_auc = roc_auc_score(train_targets, train_preds)
                val_auc = roc_auc_score(val_targets, val_preds)
                
                scheduler.step(val_auc)
                
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    patience_counter = 0
                    torch.save(model.state_dict(), os.path.join(args.working_dir, f"model_fold_{fold}.pt"))
                else:
                    patience_counter += 1
                
                if patience_counter >= args.max_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            
            # Load best model for this fold
            model.load_state_dict(torch.load(os.path.join(args.working_dir, f"model_fold_{fold}.pt")))
            models.append(model)
            
            # Make validation predictions
            val_preds = predict(model, val_dataset, args.val_batch_size, args.num_workers, device)
            oof_preds[val_idx] = val_preds
            fold_auc = roc_auc_score(y_val_fold, val_preds)
            fold_scores.append(fold_auc)
            print(f"Fold {fold} AUC: {fold_auc:.6f}")
            
            # Make test predictions for this fold
            test_dataset = TabularDataset(X_test, cat_cols, cont_cols)
            test_preds_fold = predict(model, test_dataset, args.val_batch_size, args.num_workers, device)
            test_preds += test_preds_fold / args.n_folds
        
        # Overall validation score
        overall_auc = roc_auc_score(y, oof_preds)
        print(f"\n{'='*50}")
        print(f"Cross-Validation Results:")
        print(f"Fold AUC scores: {[f'{score:.6f}' for score in fold_scores]}")
        print(f"Mean Fold AUC: {np.mean(fold_scores):.6f}")
        print(f"Std Fold AUC: {np.std(fold_scores):.6f}")
        print(f"Overall OOF AUC: {overall_auc:.6f}")
        
        # Train final model on full data
        print("\nTraining final model on full dataset...")
        full_dataset = TabularDataset(X, cat_cols, cont_cols, y)
        full_loader = DataLoader(
            full_dataset, batch_size=args.batch_size, shuffle=True, 
            num_workers=args.num_workers
        )
        
        final_model = TabularAttentionNN(cat_dims, n_cont, args.hidden_dim, args.dropout).to(device)
        optimizer = optim.AdamW(final_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion = nn.BCELoss()
        
        for epoch in range(args.final_epochs):
            for cat_batch, cont_batch, target_batch in tqdm(
                full_loader, desc=f"Final Training Epoch {epoch+1}", leave=False
            ):
                cat_batch, cont_batch, target_batch = (
                    cat_batch.to(device),
                    cont_batch.to(device),
                    target_batch.to(device),
                )
                
                optimizer.zero_grad()
                preds = final_model(cat_batch, cont_batch)
                loss = criterion(preds, target_batch)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(final_model.parameters(), 1.0)
                optimizer.step()
        
        # Create ensemble predictions with final model
        print("\nCreating ensemble predictions...")
        test_dataset = TabularDataset(X_test, cat_cols, cont_cols)
        final_test_preds = predict(final_model, test_dataset, args.val_batch_size, args.num_workers, device)
        
        # Weighted ensemble of CV models and final model
        test_preds = test_preds * 0.7 + final_test_preds * 0.3
        
        # Clean up temporary files
        for fold in range(1, args.n_folds + 1):
            model_path = os.path.join(args.working_dir, f"model_fold_{fold}.pt")
            if os.path.exists(model_path):
                os.remove(model_path)
    
    # Clip predictions to valid probability range [0, 1]
    test_preds = np.clip(test_preds, 0, 1)
    
    # Create submission file
    os.makedirs(args.output_dir, exist_ok=True)
    submission = pd.DataFrame({"id": test_ids, "target": test_preds})
    submission_path = os.path.join(args.output_dir, "submission.csv")
    submission.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
    print(f"Submission shape: {submission.shape}")
    
    # Verify submission format matches sample
    sample_submission = pd.read_csv(os.path.join(args.input_dir, "sample_submission.csv"))
    print(f"\nVerifying submission format...")
    print(f"Sample submission columns: {sample_submission.columns.tolist()}")
    print(f"Our submission columns: {submission.columns.tolist()}")
    print(f"Column match: {list(submission.columns) == list(sample_submission.columns)}")
    print(
        f"ID range match: {submission['id'].min() == sample_submission['id'].min()} and {submission['id'].max() == sample_submission['id'].max()}"
    )
    print(
        f"Target value range: [{submission['target'].min():.6f}, {submission['target'].max():.6f}]"
    )
    
    # Check if file exists and is valid
    if os.path.exists(submission_path):
        print(
            f"\n✓ Submission file successfully created at {os.path.abspath(submission_path)}"
        )
        submission_check = pd.read_csv(submission_path)
        print(f"✓ Submission file contains {len(submission_check)} rows")
        print(f"✓ First few rows of submission:")
        print(submission_check.head())
    else:
        print(f"\n✗ ERROR: Submission file not found at {submission_path}")
    
    print("\nDone!")
    print(f"Evaluation metric (AUC): {overall_auc:.6f}")


if __name__ == "__main__":
    main()