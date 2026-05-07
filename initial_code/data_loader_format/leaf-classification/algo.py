import os
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import log_loss


def set_seed(seed=42):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class LeafDataset(Dataset):
    """PyTorch Dataset for leaf classification."""
    
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long) if y is not None else None
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        else:
            return self.X[idx]


class LeafModel(nn.Module):
    """Multi-layer classifier for leaf classification."""
    
    def __init__(self, input_dim, num_classes, dropout1=0.4, dropout2=0.3, dropout3=0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.drop1 = nn.Dropout(dropout1)

        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.drop2 = nn.Dropout(dropout2)

        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.drop3 = nn.Dropout(dropout3)

        self.out = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.drop1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.drop2(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.drop3(x)

        x = self.out(x)
        return x


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Leaf Classification Training Script')
    
    # Model hyperparameters
    parser.add_argument('--dropout1', type=float, default=0.4, help='Dropout rate for first layer')
    parser.add_argument('--dropout2', type=float, default=0.3, help='Dropout rate for second layer')
    parser.add_argument('--dropout3', type=float, default=0.2, help='Dropout rate for third layer')
    
    # Training hyperparameters
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for optimizer')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--max_epochs', type=int, default=200, help='Maximum number of epochs')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    
    # Scheduler parameters
    parser.add_argument('--lr_factor', type=float, default=0.5, help='Factor for learning rate reduction')
    parser.add_argument('--lr_patience', type=int, default=5, help='Patience for learning rate scheduler')
    
    # Data parameters
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    
    # Path parameters
    parser.add_argument('--output_dir', type=str, default='./submission', help='Output directory for submission')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize data loader and get data
    data_loader = MyDataLoader()
    train_data, test_data = data_loader.get_data()
    
    # Extract data
    X_train = train_data['X_train']
    y_train = train_data['y_train']
    X_val = train_data['X_val']
    y_val = train_data['y_val']
    class_weights = train_data['class_weights']
    num_classes = train_data['num_classes']
    class_names = train_data['class_names']
    
    X_test = test_data['X_test']
    test_ids = test_data['test_ids']
    
    # Create datasets and dataloaders
    train_dataset = LeafDataset(X_train, y_train)
    val_dataset = LeafDataset(X_val, y_val)
    test_dataset = LeafDataset(X_test)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    
    # Create model
    input_dim = X_train.shape[1]
    model = LeafModel(input_dim=input_dim, num_classes=num_classes,
                      dropout1=args.dropout1, dropout2=args.dropout2, dropout3=args.dropout3)
    model = model.to(device)
    
    # Loss, optimizer, scheduler
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=args.lr_factor, patience=args.lr_patience
    )
    
    # Training with early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    
    for epoch in range(args.max_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        train_loss /= len(train_loader.dataset)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                probs = torch.softmax(outputs, dim=1)
                val_preds.append(probs.cpu().numpy())
                val_labels.append(y_batch.cpu().numpy())
        val_loss /= len(val_loader.dataset)
        val_preds = np.vstack(val_preds)
        val_labels = np.concatenate(val_labels)
        val_log = log_loss(val_labels, val_preds, labels=np.arange(num_classes))
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1:3d}/{args.max_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val LogLoss: {val_log:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("Early stopping triggered.")
                break
    
    # Load best model
    model.load_state_dict(best_state)
    model.eval()
    
    # Compute final validation metric
    val_preds = []
    val_labels = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            probs = torch.softmax(outputs, dim=1)
            val_preds.append(probs.cpu().numpy())
            val_labels.append(y_batch.cpu().numpy())
    val_preds = np.vstack(val_preds)
    val_labels = np.concatenate(val_labels)
    final_val_log = log_loss(val_labels, val_preds, labels=np.arange(num_classes))
    print(f"\nFinal Validation Log Loss: {final_val_log:.5f}")
    
    # Generate test predictions
    test_preds = []
    with torch.no_grad():
        for X_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            probs = torch.softmax(outputs, dim=1)
            test_preds.append(probs.cpu().numpy())
    test_preds = np.vstack(test_preds)
    
    # Clip and renormalize (safeguard for log)
    test_preds = np.clip(test_preds, 1e-15, 1-1e-15)
    test_preds = test_preds / test_preds.sum(axis=1, keepdims=True)
    
    # Create submission dataframe
    submission = pd.DataFrame(test_preds, columns=class_names)
    submission.insert(0, 'id', test_ids)
    
    # Ensure submission directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    submission.to_csv(os.path.join(args.output_dir, 'submission.csv'), index=False)
    print(f"Submission saved to {os.path.join(args.output_dir, 'submission.csv')}")
    
    print("Done.")


if __name__ == "__main__":
    main()