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