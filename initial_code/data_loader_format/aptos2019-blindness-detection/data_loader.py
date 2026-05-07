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