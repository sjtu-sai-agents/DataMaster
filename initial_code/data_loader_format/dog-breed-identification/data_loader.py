import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class DogDataset(Dataset):
    """Custom Dataset for dog breed classification."""
    
    def __init__(self, ids, labels, img_dir, transform=None, is_train=True):
        self.ids = ids.values if hasattr(ids, "values") else ids
        self.labels = labels.values if hasattr(labels, "values") else labels
        self.img_dir = img_dir
        self.transform = transform
        self.is_train = is_train

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, f"{self.ids[idx]}.jpg")
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.is_train:
            label = self.labels[idx]
            return image, label
        else:
            return image


class MyDataLoader(BaseDataLoader):
    """Data loader for dog breed classification with data augmentation."""
    
    def __init__(self, batch_size=32, num_workers=8, input_dir="./input/", **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.input_dir = input_dir
        self.label_encoder = None
        self.num_classes = None
        self.train_transform = None
        self.val_transform = None

    def setup(self):
        """
        Load data, perform feature engineering, and create data loaders.
        Uses fixed validation set from input/val.csv if available.
        """
        # Load labels
        labels_df = pd.read_csv(os.path.join(self.input_dir, "labels.csv"))
        
        # Encode labels
        le = LabelEncoder()
        labels_df["breed_encoded"] = le.fit_transform(labels_df["breed"])
        self.label_encoder = le
        self.num_classes = len(le.classes_)
        
        # Check for fixed validation set
        val_csv_path = os.path.join(self.input_dir, "val.csv")
        if os.path.exists(val_csv_path):
            # Use fixed validation set - strictly required
            val_df = pd.read_csv(val_csv_path)
            # Handle both 'id' and 'image' column names
            if 'id' in val_df.columns:
                val_ids_set = set(val_df['id'].values)
            elif 'image' in val_df.columns:
                val_ids_set = set(val_df['image'].values)
            else:
                raise ValueError("val.csv must have 'id' or 'image' column")
            
            # Split train and validation
            train_df = labels_df[~labels_df['id'].isin(val_ids_set)]
            val_df_merged = labels_df[labels_df['id'].isin(val_ids_set)]
            
            train_ids = train_df['id']
            train_labels = train_df['breed_encoded']
            val_ids = val_df_merged['id']
            val_labels = val_df_merged['breed_encoded']
        else:
            # Fallback to stratified split if no val.csv exists
            train_ids, val_ids, train_labels, val_labels = train_test_split(
                labels_df["id"],
                labels_df["breed_encoded"],
                test_size=0.2,
                random_state=42,
                stratify=labels_df["breed_encoded"],
            )
        
        # Define data transforms with augmentation
        self.train_transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.RandomResizedCrop(320, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.CenterCrop(320),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Create datasets
        train_dataset = DogDataset(
            train_ids, train_labels, 
            os.path.join(self.input_dir, "train/"), 
            transform=self.train_transform, is_train=True
        )
        val_dataset = DogDataset(
            val_ids, val_labels, 
            os.path.join(self.input_dir, "train/"), 
            transform=self.val_transform, is_train=True
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True
        )
        
        # Prepare test data
        test_dir = os.path.join(self.input_dir, "test/")
        test_ids = [os.path.splitext(f)[0] for f in os.listdir(test_dir) if f.endswith(".jpg")]
        test_dataset = DogDataset(
            test_ids, None, test_dir,
            transform=self.val_transform, is_train=False
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True
        )
        
        # Set train_data and test_data
        self.train_data = {
            'train_loader': train_loader,
            'val_loader': val_loader,
            'num_classes': self.num_classes,
            'label_encoder': self.label_encoder
        }
        self.test_data = {
            'test_loader': test_loader,
            'test_ids': test_ids,
            'label_encoder': self.label_encoder
        }

    def describe(self) -> str:
        """
        Return a description of the data processing approach.
        """
        desc = "Dog Breed Classification DataLoader:\n"
        desc += "- Uses EfficientNet-B3 compatible image transforms (384x384 resize, 320x320 crop)\n"
        desc += "- Training augmentation: RandomResizedCrop, RandomHorizontalFlip, RandomRotation, ColorJitter\n"
        desc += "- Validation: CenterCrop without augmentation\n"
        desc += f"- Number of classes: {self.num_classes if self.num_classes else 'Not set'}\n"
        desc += "- Uses fixed validation set from input/val.csv if available (strict requirement)\n"
        desc += "- Falls back to stratified 80/20 split if val.csv not found"
        return desc