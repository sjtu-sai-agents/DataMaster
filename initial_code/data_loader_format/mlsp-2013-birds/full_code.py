import os
import warnings
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


# Configuration constants
DATA_ROOT = "./input"
ESSENTIAL = os.path.join(DATA_ROOT, "essential_data")
SUPPLEMENTAL = os.path.join(DATA_ROOT, "supplemental_data")
SPECTROGRAMS_DIR = os.path.join(SUPPLEMENTAL, "filtered_spectrograms")


class BirdSpectrogramDataset(Dataset):
    """Dataset for bird spectrogram multi-label classification."""
    
    def __init__(self, items, img_size=224, num_classes=19, spectrograms_dir=None):
        self.items = items
        self.img_size = img_size
        self.num_classes = num_classes
        self.spectrograms_dir = spectrograms_dir or SPECTROGRAMS_DIR

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        rec_id = item['rec_id']
        base = item['base']
        labels = item['labels']

        # Load and preprocess spectrogram
        img_path = os.path.join(self.spectrograms_dir, f"{base}.bmp")
        if os.path.exists(img_path):
            try:
                img = Image.open(img_path).convert('L')   # grayscale

                # Convert to tensor [0,1]
                img_tensor = torch.from_numpy(np.array(img)).float().div(255.0)

                # Per‑image standardization
                mean = img_tensor.mean()
                std = img_tensor.std()
                if std > 1e-6:
                    img_tensor = (img_tensor - mean) / std
                else:
                    img_tensor = img_tensor - mean

                # Expand to 3 channels
                img_tensor = img_tensor.unsqueeze(0)          # (1, H, W)
                img_tensor = img_tensor.repeat(3, 1, 1)       # (3, H, W)

                # Resize
                resize = transforms.Resize((self.img_size, self.img_size), antialias=True)
                img_tensor = resize(img_tensor)

                # Normalize with ImageNet statistics
                normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])
                img_tensor = normalize(img_tensor)

            except Exception as e:
                warnings.warn(f"Failed to load {img_path}: {e}, using zero tensor")
                img_tensor = torch.zeros((3, self.img_size, self.img_size), dtype=torch.float)
        else:
            warnings.warn(f"Spectrogram not found: {img_path}, using zero tensor")
            img_tensor = torch.zeros((3, self.img_size, self.img_size), dtype=torch.float)

        # Multi‑hot label vector
        label_vec = torch.zeros(self.num_classes, dtype=torch.float)
        for s in labels:
            if 0 <= s < self.num_classes:
                label_vec[s] = 1.0

        return img_tensor, label_vec, rec_id


class MyDataLoader(BaseDataLoader):
    """Data loader for bird spectrogram classification."""
    
    def __init__(self, img_size=224, num_classes=19, batch_size=32, num_workers=4, 
                 data_root="./input", random_seed=42, **kwargs):
        super().__init__(**kwargs)
        self.img_size = img_size
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_root = data_root
        self.random_seed = random_seed
        
        # Update paths based on data_root
        self.essential_dir = os.path.join(data_root, "essential_data")
        self.supplemental_dir = os.path.join(data_root, "supplemental_data")
        self.spectrograms_dir = os.path.join(self.supplemental_dir, "filtered_spectrograms")
        
        self.mapping = {}
        self.train_items = []
        self.val_items = []
        self.test_items = []
        
    def _parse_mapping(self):
        """Parse mapping: rec_id -> base filename (without .wav)"""
        mapping = {}
        mapping_file = os.path.join(self.essential_dir, "rec_id2filename.txt")
        with open(mapping_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = [p.strip().strip('"') for p in line.split(",")]
                try:
                    rec_id = int(parts[0])          # skip header if conversion fails
                except ValueError:
                    continue
                wav_name = parts[1]
                base = os.path.splitext(wav_name)[0]   # remove .wav
                mapping[rec_id] = base
        return mapping
    
    def _parse_labels(self):
        """Parse training labels from rec_labels_test_hidden.txt"""
        labeled_items = []
        labels_file = os.path.join(self.essential_dir, "rec_labels_test_hidden.txt")
        with open(labels_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if "?" in line:                     # skip test recordings
                    continue
                parts = line.split(",")
                try:
                    rec_id = int(parts[0])          # skip header if conversion fails
                except ValueError:
                    continue
                labels = []
                for s in parts[1:]:
                    if s:
                        try:
                            labels.append(int(s))
                        except ValueError:
                            pass
                if rec_id in self.mapping:
                    labeled_items.append({
                        'rec_id': rec_id,
                        'base': self.mapping[rec_id],
                        'labels': labels
                    })
                else:
                    warnings.warn(f"rec_id {rec_id} not in mapping, skipping")
        return labeled_items
    
    def setup(self):
        """
        Load data, feature engineering, data augmentation, etc.
        Must set self.train_data and self.test_data
        """
        # Parse mapping
        self.mapping = self._parse_mapping()
        
        # Parse labeled items
        labeled_items = self._parse_labels()
        
        print(f"Number of labeled training recordings: {len(labeled_items)}")
        if len(labeled_items) == 0:
            raise RuntimeError("No training items found. Check file paths and parsing.")
        
        # Split into training and validation using fixed val.csv
        val_csv_path = os.path.join(self.data_root, 'val.csv')
        if os.path.exists(val_csv_path):
            val_df = pd.read_csv(val_csv_path)
            # Check for 'rec_id' column, fallback to 'image' or first column
            if 'rec_id' in val_df.columns:
                val_rec_ids = set(val_df['rec_id'].values)
            elif 'image' in val_df.columns:
                val_rec_ids = set(val_df['image'].values)
            else:
                val_rec_ids = set(val_df.iloc[:, 0].values)
            
            self.train_items = [item for item in labeled_items if item['rec_id'] not in val_rec_ids]
            self.val_items = [item for item in labeled_items if item['rec_id'] in val_rec_ids]
            print(f"Using fixed validation set from val.csv: {len(self.val_items)} validation samples")
        else:
            # Fallback: use train_test_split if val.csv not found
            from sklearn.model_selection import train_test_split
            self.train_items, self.val_items = train_test_split(
                labeled_items, test_size=0.2, random_state=self.random_seed
            )
            print(f"Warning: val.csv not found, using random 80/20 split")
        
        print(f"Training samples: {len(self.train_items)}, Validation samples: {len(self.val_items)}")
        
        # Build test items from sample_submission
        sub_sample = pd.read_csv(os.path.join(self.data_root, "sample_submission.csv"))
        sub_sample['rec_id'] = sub_sample['Id'].astype(int) // 100
        test_rec_ids = sub_sample['rec_id'].unique()
        
        self.test_items = []
        for rec_id in test_rec_ids:
            if rec_id in self.mapping:
                self.test_items.append({
                    'rec_id': rec_id,
                    'base': self.mapping[rec_id],
                    'labels': []          # dummy
                })
            else:
                warnings.warn(f"Test rec_id {rec_id} not in mapping, skipping")
        
        print(f"Number of test recordings: {len(self.test_items)}")
        
        # Create datasets
        train_dataset = BirdSpectrogramDataset(
            self.train_items, 
            img_size=self.img_size, 
            num_classes=self.num_classes,
            spectrograms_dir=self.spectrograms_dir
        )
        val_dataset = BirdSpectrogramDataset(
            self.val_items, 
            img_size=self.img_size, 
            num_classes=self.num_classes,
            spectrograms_dir=self.spectrograms_dir
        )
        test_dataset = BirdSpectrogramDataset(
            self.test_items, 
            img_size=self.img_size, 
            num_classes=self.num_classes,
            spectrograms_dir=self.spectrograms_dir
        )
        
        # Create data loaders
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
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        self.train_data = train_loader
        self.test_data = {
            'val_loader': val_loader,
            'test_loader': test_loader,
            'sub_sample': sub_sample
        }
        
    def describe(self) -> str:
        """
        Return a description of your data processing approach
        """
        return """
        Bird Spectrogram Data Loader for Multi-label Classification
        
        Data Processing:
        - Loads spectrogram images from filtered_spectrograms directory
        - Converts grayscale images to 3-channel tensors
        - Applies per-image standardization
        - Resizes to 224x224 for ResNet compatibility
        - Normalizes with ImageNet statistics
        
        Data Split:
        - Uses fixed validation set from input/val.csv if available
        - Falls back to 80/20 random split if val.csv not found
        - Test data from sample_submission.csv
        
        Features:
        - Multi-label classification (19 bird species)
        - Handles missing spectrograms with zero tensors
        """

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