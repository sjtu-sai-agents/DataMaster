import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from sklearn.metrics import roc_auc_score


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Cactus Classification Training')
    
    # Model hyperparameters
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate (default: 0.0001)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Image size (default: 224)')
    parser.add_argument('--max_epochs', type=int, default=15,
                        help='Maximum number of epochs (default: 15)')
    parser.add_argument('--patience', type=int, default=3,
                        help='Early stopping patience (default: 3)')
    
    # Path parameters
    parser.add_argument('--output_dir', type=str, default='./submission',
                        help='Output directory for submissions and models (default: ./submission)')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for dataloader (default: 4)')
    
    return parser.parse_args()


def set_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    """Main training function."""
    args = parse_args()
    
    # Set random seeds for reproducibility
    set_seed(args.seed)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize data loader
    data_loader = MyDataLoader(
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.num_workers,
        seed=args.seed
    )
    
    # Get data
    train_data, test_data = data_loader.get_data()
    
    train_loader = train_data['train_loader']
    val_loader = train_data['val_loader']
    test_loaders = test_data['test_loaders']
    test_ids = test_data['test_ids']
    
    # Model - ResNet18 with pretrained ImageNet weights
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model = model.to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop with early stopping
    best_auc = 0.0
    best_epoch = 0
    patience_counter = 0
    best_model_path = os.path.join(args.output_dir, "best_model.pth")
    
    print("\nStarting training...")
    for epoch in range(args.max_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).view(-1, 1)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        train_loss /= len(train_loader.dataset)
        
        # Validation phase
        model.eval()
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                outputs = model(images)
                probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                val_preds.extend(probs)
                val_labels.extend(labels.cpu().numpy())
        
        auc = roc_auc_score(val_labels, val_preds)
        print(f"Epoch {epoch+1}/{args.max_epochs} - Train Loss: {train_loss:.4f} - Val AUC: {auc:.4f}")
        
        # Early stopping and checkpoint
        if auc > best_auc:
            best_auc = auc
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    print(f"\nBest validation AUC: {best_auc:.6f} (epoch {best_epoch+1})")
    
    # Load best model for test predictions
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    
    def predict_tta(loader):
        """Predict with TTA (Test Time Augmentation)."""
        all_probs = []
        all_ids = []
        with torch.no_grad():
            for images, ids in loader:
                images = images.to(device)
                outputs = model(images)
                probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                all_probs.extend(probs)
                all_ids.extend(ids)
        # Ensure order matches original test_df
        df = pd.DataFrame({'id': all_ids, 'prob': all_probs})
        df = df.set_index('id').reindex(test_ids).reset_index()
        return df['prob'].values
    
    # TTA predictions
    print("\nGenerating predictions with TTA...")
    probs_orig = predict_tta(test_loaders['orig'])
    probs_hflip = predict_tta(test_loaders['hflip'])
    probs_vflip = predict_tta(test_loaders['vflip'])
    
    # Average probabilities from TTA
    final_probs = (probs_orig + probs_hflip + probs_vflip) / 3.0
    
    # Create submission file
    submission = pd.DataFrame({'id': test_ids, 'has_cactus': final_probs})
    submission_path = os.path.join(args.output_dir, "submission.csv")
    submission.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")


if __name__ == "__main__":
    main()