import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class PlantDataset(Dataset):
    """Dataset class for Plant Pathology data."""
    
    def __init__(self, df, transform=None, is_test=False, image_dir="./input/images"):
        self.df = df
        self.transform = transform
        self.is_test = is_test
        self.image_dir = image_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row['image_id']
        img_path = os.path.join(self.image_dir, f"{image_id}.jpg")
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        if self.is_test:
            return image, image_id
        else:
            labels = row[['healthy', 'multiple_diseases', 'rust', 'scab']].values.astype(np.float32)
            return image, torch.from_numpy(labels)


class MyDataLoader(BaseDataLoader):
    """Data loader for Plant Pathology classification task."""
    
    def __init__(self, img_size=224, batch_size=32, num_workers=4, **kwargs):
        super().__init__(**kwargs)
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        
    def setup(self):
        """
        Load data, feature engineering, data augmentation, etc.
        Must set self.train_data and self.test_data
        """
        # Load data
        train_full_df = pd.read_csv('./input/train.csv')
        test_df = pd.read_csv('./input/test.csv')
        
        # Check if validation set exists - use fixed val.csv if available
        if os.path.exists('input/val.csv'):
            val_df = pd.read_csv('input/val.csv')
            # Handle column name compatibility (image vs image_id)
            if 'image' in val_df.columns and 'image_id' not in val_df.columns:
                val_df = val_df.rename(columns={'image': 'image_id'})
            
            # Remove val samples from train
            val_images = set(val_df['image_id'].values)
            train_df = train_full_df[~train_full_df['image_id'].isin(val_images)].reset_index(drop=True)
            
            # If val_df doesn't have label columns, get them from train_full_df
            label_cols = ['healthy', 'multiple_diseases', 'rust', 'scab']
            if not all(col in val_df.columns for col in label_cols):
                val_df = train_full_df[train_full_df['image_id'].isin(val_images)].reset_index(drop=True)
            else:
                val_df = val_df.reset_index(drop=True)
        else:
            # Fallback: use train_test_split only if val.csv doesn't exist
            from sklearn.model_selection import train_test_split
            train_df, val_df = train_test_split(
                train_full_df, test_size=0.2, random_state=42, shuffle=True
            )
            train_df = train_df.reset_index(drop=True)
            val_df = val_df.reset_index(drop=True)
        
        # Define transforms
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(self.img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        test_transform = val_transform
        
        # Create datasets
        train_ds = PlantDataset(train_df, transform=train_transform, is_test=False)
        val_ds = PlantDataset(val_df, transform=val_transform, is_test=False)
        test_ds = PlantDataset(test_df, transform=test_transform, is_test=True)
        
        # Create dataloaders
        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True
        )
        test_loader = DataLoader(
            test_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True
        )
        
        self.train_data = (train_loader, val_loader)
        self.test_data = test_loader
        
    def describe(self) -> str:
        """
        Return a description of your data processing approach
        """
        return """
        Plant Pathology Data Loader:
        - Loads train, validation, and test data from CSV files
        - Uses fixed validation set from input/val.csv if available (strictly no random split when val.csv exists)
        - Applies data augmentation for training: RandomResizedCrop, RandomHorizontalFlip, ColorJitter
        - Uses standard ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        - Returns PyTorch DataLoaders for training, validation, and testing
        - Compatible with different column naming conventions (image/image_id)
        """

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