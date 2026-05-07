import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ImageNet normalization
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class CactusDataset(Dataset):
    """Dataset class for Cactus classification."""
    
    def __init__(self, df, img_dir, transform=None, is_test=False):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.df.iloc[idx]['id']
        img_path = os.path.join(self.img_dir, img_id)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        if self.is_test:
            return image, img_id
        else:
            label = self.df.iloc[idx]['has_cactus']
            return image, torch.tensor(label, dtype=torch.float)


class MyDataLoader(BaseDataLoader):
    """DataLoader for Cactus classification task."""
    
    def __init__(self, batch_size=32, img_size=224, num_workers=4, seed=42, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_workers = num_workers
        self.seed = seed
        
        # Paths
        self.input_dir = "./input"
        self.train_img_dir = os.path.join(self.input_dir, "train")
        self.test_img_dir = os.path.join(self.input_dir, "test")
        self.train_csv = os.path.join(self.input_dir, "train.csv")
        self.sample_sub = os.path.join(self.input_dir, "sample_submission.csv")

    def setup(self):
        """
        Load data, feature engineering, data augmentation, etc.
        Must set self.train_data and self.test_data
        """
        # Load training labels
        train_df = pd.read_csv(self.train_csv)
        
        # Check if val.csv exists and use it (fixed validation set)
        val_csv_path = os.path.join(self.input_dir, "val.csv")
        if os.path.exists(val_csv_path):
            val_ids_df = pd.read_csv(val_csv_path)
            # Check which column to use for image IDs in val.csv
            if 'id' in val_ids_df.columns:
                id_col = 'id'
            elif 'image' in val_ids_df.columns:
                id_col = 'image'
            else:
                id_col = val_ids_df.columns[0]
            
            val_image_ids = set(val_ids_df[id_col].values)
            # Get validation samples from full train_df (with labels)
            val_df = train_df[train_df['id'].isin(val_image_ids)].reset_index(drop=True)
            # Remove validation samples from training set
            train_df_sub = train_df[~train_df['id'].isin(val_image_ids)].reset_index(drop=True)
            print(f"Using fixed validation set from val.csv: {len(val_df)} samples")
        else:
            # Split from original competition data only
            from sklearn.model_selection import train_test_split
            train_ids, val_ids = train_test_split(
                train_df.index,
                test_size=0.1,
                stratify=train_df['has_cactus'],
                random_state=self.seed
            )
            train_df_sub = train_df.loc[train_ids].reset_index(drop=True)
            val_df = train_df.loc[val_ids].reset_index(drop=True)
            print(f"Using stratified split: {len(train_df_sub)} train, {len(val_df)} val")
        
        print(f"Training samples: {len(train_df_sub)}")
        print(f"Validation samples: {len(val_df)}")
        
        # Define transforms
        train_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])
        
        # Test transforms for TTA
        test_transform_orig = val_transform
        test_transform_hflip = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])
        test_transform_vflip = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.RandomVerticalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])
        
        # Create datasets
        train_dataset = CactusDataset(train_df_sub, self.train_img_dir, train_transform)
        val_dataset = CactusDataset(val_df, self.train_img_dir, val_transform)
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        # Load test data
        sub_df = pd.read_csv(self.sample_sub)
        test_ids = sub_df['id'].tolist()
        test_df = pd.DataFrame({'id': test_ids})
        
        # Create test datasets for TTA
        test_dataset_orig = CactusDataset(test_df, self.test_img_dir, test_transform_orig, is_test=True)
        test_dataset_hflip = CactusDataset(test_df, self.test_img_dir, test_transform_hflip, is_test=True)
        test_dataset_vflip = CactusDataset(test_df, self.test_img_dir, test_transform_vflip, is_test=True)
        
        test_loader_orig = DataLoader(
            test_dataset_orig,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        test_loader_hflip = DataLoader(
            test_dataset_hflip,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        test_loader_vflip = DataLoader(
            test_dataset_vflip,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        # Set train_data and test_data
        self.train_data = {
            'train_loader': train_loader,
            'val_loader': val_loader,
            'train_df': train_df_sub,
            'val_df': val_df
        }
        
        self.test_data = {
            'test_loaders': {
                'orig': test_loader_orig,
                'hflip': test_loader_hflip,
                'vflip': test_loader_vflip
            },
            'test_ids': test_ids
        }

    def describe(self) -> str:
        """
        Return a description of your data processing approach
        """
        return """
        Data processing approach for Cactus classification:
        - Image resizing to 224x224
        - Data augmentation: RandomHorizontalFlip, RandomRotation(10 degrees)
        - ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        - Test Time Augmentation (TTA): original, horizontal flip, vertical flip
        - Uses fixed validation set from input/val.csv if available
        - Falls back to stratified 90/10 split if val.csv not found
        """

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