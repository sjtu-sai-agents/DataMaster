import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from sklearn.metrics import roc_auc_score


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Bird Spectrogram Multi-label Classification')
    
    # Model hyperparameters
    parser.add_argument('--num_classes', type=int, default=19,
                        help='Number of output classes (default: 19)')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Input image size (default: 224)')
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training (default: 32)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers (default: 4)')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs (default: 30)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 1e-3)')
    parser.add_argument('--lr_step_size', type=int, default=5,
                        help='Learning rate decay step size (default: 5)')
    parser.add_argument('--lr_gamma', type=float, default=0.1,
                        help='Learning rate decay gamma (default: 0.1)')
    
    # Path parameters
    parser.add_argument('--data_root', type=str, default='./input',
                        help='Root directory of data (default: ./input)')
    parser.add_argument('--output_dir', type=str, default='submission',
                        help='Output directory for submission (default: submission)')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (default: cuda)')
    
    return parser.parse_args()


def main():
    """Main training and inference function."""
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create data loader
    data_loader = MyDataLoader(
        img_size=args.img_size,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_root=args.data_root,
        random_seed=args.seed
    )
    
    # Get data
    train_loader, test_data = data_loader.get_data()
    val_loader = test_data['val_loader']
    test_loader = test_data['test_loader']
    sub_sample = test_data['sub_sample']
    
    # Build model
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, args.num_classes)
    model = model.to(device)
    
    # Loss function
    criterion = nn.BCEWithLogitsLoss()
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=args.lr_step_size, 
        gamma=args.lr_gamma
    )
    
    # Training loop
    best_val_auc = 0.0
    best_model_state = None
    
    print(f"\nStarting training for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for images, labels, _ in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
        
        train_loss /= len(train_loader.dataset)
        
        # Validation phase
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                probs = torch.sigmoid(outputs)
                all_preds.append(probs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        if all_preds:
            all_preds = np.vstack(all_preds)
            all_labels = np.vstack(all_labels)
            val_auc = roc_auc_score(all_labels.ravel(), all_preds.ravel())
        else:
            val_auc = 0.5
        
        print(f"Epoch {epoch+1}/{args.epochs} - Train loss: {train_loss:.4f} - Val AUC: {val_auc:.4f}")
        
        # Update learning rate
        scheduler.step()
        
        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict().copy()
    
    # Load best model for inference
    model.load_state_dict(best_model_state)
    print(f"\nBest validation AUC: {best_val_auc:.4f}")
    
    # Inference on test set
    print("\nRunning inference on test set...")
    model.eval()
    pred_dict = {}
    
    with torch.no_grad():
        for images, _, rec_ids in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy()
            for i, rec_id in enumerate(rec_ids):
                rec_id = rec_id.item()
                pred_dict[rec_id] = probs[i, :]
    
    # Create submission file
    submission_df = sub_sample.copy()
    submission_df['Probability'] = 0.0
    
    for idx, row in submission_df.iterrows():
        rec_id = row['rec_id']
        species = int(row['Id']) % 100
        if rec_id in pred_dict:
            submission_df.at[idx, 'Probability'] = pred_dict[rec_id][species]
        else:
            submission_df.at[idx, 'Probability'] = 0.0
    
    # Keep only required columns
    submission_df = submission_df[['Id', 'Probability']]
    
    # Save submission
    os.makedirs(args.output_dir, exist_ok=True)
    submission_path = os.path.join(args.output_dir, 'submission.csv')
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")


if __name__ == "__main__":
    main()