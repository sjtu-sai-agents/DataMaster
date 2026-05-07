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