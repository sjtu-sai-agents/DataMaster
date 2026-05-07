import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from sklearn.metrics import roc_auc_score
import argparse


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Plant Pathology Training Script')
    parser.add_argument('--img_size', type=int, default=224, 
                        help='Input image size (default: 224)')
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='Batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=15, 
                        help='Number of training epochs (default: 15)')
    parser.add_argument('--lr', type=float, default=0.001, 
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--t_max', type=int, default=10, 
                        help='T_max for cosine annealing scheduler (default: 10)')
    parser.add_argument('--num_workers', type=int, default=4, 
                        help='Number of workers for data loading (default: 4)')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--output_dir', type=str, default='./working', 
                        help='Output directory for saving models (default: ./working)')
    parser.add_argument('--submission_dir', type=str, default='./submission', 
                        help='Directory for saving submission files (default: ./submission)')
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize data loader and get data
    data_loader = MyDataLoader(
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    train_data, test_loader = data_loader.get_data()
    train_loader, val_loader = train_data
    
    print(f"\nData loader description:{data_loader.describe()}")
    
    # Model setup - EfficientNet B0
    model = models.efficientnet_b0(pretrained=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 4)
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.t_max)
    
    # Training loop
    best_auc = 0.0
    best_state = None
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\nStarting training for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
        
        train_loss /= len(train_loader.dataset)
        scheduler.step()

        # Validation phase
        model.eval()
        val_loss = 0.0
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                probs = torch.sigmoid(outputs)
                all_labels.append(labels.cpu().numpy())
                all_probs.append(probs.cpu().numpy())
        
        val_loss /= len(val_loader.dataset)
        all_labels = np.concatenate(all_labels, axis=0)
        all_probs = np.concatenate(all_probs, axis=0)

        # Calculate AUC for each class
        auc_scores = []
        for i in range(4):
            try:
                auc = roc_auc_score(all_labels[:, i], all_probs[:, i])
            except ValueError:
                auc = 0.5
            auc_scores.append(auc)
        mean_auc = np.mean(auc_scores)

        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {mean_auc:.4f}")

        # Save best model
        if mean_auc > best_auc:
            best_auc = mean_auc
            best_state = model.state_dict().copy()
            torch.save(best_state, os.path.join(args.output_dir, "best_model.pth"))

    # Load best model for inference
    model.load_state_dict(best_state)
    print(f"\nBest Validation AUC: {best_auc:.4f}")

    # Test prediction
    model.eval()
    test_ids = []
    test_preds = []
    
    with torch.no_grad():
        for images, ids in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy()
            test_preds.append(probs)
            test_ids.extend(ids)
    
    test_preds = np.concatenate(test_preds, axis=0)

    # Create submission file
    sub_df = pd.DataFrame({
        'image_id': test_ids,
        'healthy': test_preds[:, 0],
        'multiple_diseases': test_preds[:, 1],
        'rust': test_preds[:, 2],
        'scab': test_preds[:, 3]
    })
    sub_df = sub_df[['image_id', 'healthy', 'multiple_diseases', 'rust', 'scab']]

    os.makedirs(args.submission_dir, exist_ok=True)
    submission_path = os.path.join(args.submission_dir, "submission.csv")
    sub_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")


if __name__ == "__main__":
    main()