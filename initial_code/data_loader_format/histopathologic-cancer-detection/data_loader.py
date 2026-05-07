import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split


class HistopathologyDataset(Dataset):
    """Dataset for histopathology images with labels."""
    
    def __init__(self, dataframe, image_dir, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_id = self.dataframe.loc[idx, 'id']
        label = self.dataframe.loc[idx, 'label']
        img_path = os.path.join(self.image_dir, f"{img_id}.tif")
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(label, dtype=torch.float32)
        return image, label


class TestDataset(Dataset):
    """Dataset for test images without labels."""
    
    def __init__(self, dataframe, image_dir, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_id = self.dataframe.loc[idx, 'id']
        img_path = os.path.join(self.image_dir, f"{img_id}.tif")
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image, img_id


class MyDataLoader(BaseDataLoader):
    """
    Custom DataLoader for histopathology cancer detection.
    Handles data loading, preprocessing, and augmentation.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Paths
        self.input_dir = kwargs.get('input_dir', './input')
        self.train_dir = os.path.join(self.input_dir, "train")
        self.test_dir = os.path.join(self.input_dir, "test")
        self.train_labels_path = os.path.join(self.input_dir, "train_labels.csv")
        self.sample_sub_path = os.path.join(self.input_dir, "sample_submission.csv")
        
    def setup(self):
        """
        Load data, feature engineering, data augmentation, etc.
        Must set self.train_data and self.test_data
        """
        # Load labels
        df = pd.read_csv(self.train_labels_path)
        
        # Check if val.csv exists - use fixed validation set
        val_csv_path = os.path.join(self.input_dir, "val.csv")
        if os.path.exists(val_csv_path):
            val_df = pd.read_csv(val_csv_path)
            val_images = set(val_df['id'].values)
            train_df = df[~df['id'].isin(val_images)].reset_index(drop=True)
            print(f"Using fixed validation set from val.csv")
        else:
            # If no val.csv, split from original competition data only
            train_df, val_df = train_test_split(
                df, test_size=0.1, stratify=df['label'], random_state=42
            )
            train_df = train_df.reset_index(drop=True)
            val_df = val_df.reset_index(drop=True)
            print(f"Created validation split from training data (10% stratified)")
        
        print(f"Train size: {len(train_df)}, Val size: {len(val_df)}")
        
        # Test IDs from sample submission
        sample_sub = pd.read_csv(self.sample_sub_path)
        test_ids = sample_sub['id']
        test_df = pd.DataFrame({'id': test_ids})
        print(f"Test size: {len(test_df)}")
        
        # Transforms for training (with augmentation)
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        
        # Transforms for validation/test (no augmentation)
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        
        test_transform = val_transform
        
        # Create datasets
        train_dataset = HistopathologyDataset(train_df, self.train_dir, transform=train_transform)
        val_dataset = HistopathologyDataset(val_df, self.train_dir, transform=val_transform)
        test_dataset = TestDataset(test_df, self.test_dir, transform=test_transform)
        
        # Store data
        self.train_data = {
            'train_dataset': train_dataset,
            'val_dataset': val_dataset,
            'val_labels': val_df['label'].values
        }
        self.test_data = {
            'test_dataset': test_dataset,
            'test_ids': test_ids
        }
        
    def describe(self) -> str:
        """
        Return a description of your data processing approach
        """
        return """
        Histopathology Cancer Detection DataLoader:
        
        Data Processing:
        - Loads histopathology images from train/test directories
        - Uses fixed validation set from input/val.csv if available
        - Falls back to stratified 10% split if val.csv doesn't exist
        
        Data Augmentation (Training only):
        - Resize to 224x224 for EfficientNet-B3
        - RandomHorizontalFlip
        - RandomVerticalFlip  
        - RandomRotation(20 degrees)
        
        Normalization:
        - ImageNet mean: [0.485, 0.456, 0.406]
        - ImageNet std: [0.229, 0.224, 0.225]
        """