import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
import warnings

warnings.filterwarnings("ignore")

TARGET_COLS = [
    "ETT - Abnormal",
    "ETT - Borderline",
    "ETT - Normal",
    "NGT - Abnormal",
    "NGT - Borderline",
    "NGT - Incompletely Imaged",
    "NGT - Normal",
    "CVC - Abnormal",
    "CVC - Borderline",
    "CVC - Normal",
    "Swan Ganz Catheter Present",
]


class ChestXRayDataset(Dataset):
    """Dataset class for Chest X-Ray images."""
    
    def __init__(self, df, img_dir, transform=None, target_cols=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.target_cols = target_cols if target_cols else TARGET_COLS
        self.has_targets = all(col in df.columns for col in self.target_cols)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, f"{row['StudyInstanceUID']}.jpg")
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.has_targets:
            labels = row[self.target_cols].values.astype(np.float32)
        else:
            labels = np.zeros(len(self.target_cols), dtype=np.float32)

        return image, torch.tensor(labels)


class MyDataLoader(BaseDataLoader):
    """Data loader for Chest X-Ray Catheter and Line Position Classification."""
    
    def __init__(self, data_dir="./input", img_size=384, folds=5, seed=42, **kwargs):
        super().__init__(**kwargs)
        self.data_dir = data_dir
        self.img_size = img_size
        self.folds = folds
        self.seed = seed
        self.target_cols = TARGET_COLS
        
        torch.manual_seed(seed)
        np.random.seed(seed)

    def setup(self):
        """
        Load data, create folds, and define transformations.
        Uses fixed validation set from val.csv if available, otherwise uses GroupKFold.
        """
        # Load training data
        train_csv = os.path.join(self.data_dir, "train.csv")
        train_df = pd.read_csv(train_csv)
        train_dir = os.path.join(self.data_dir, "train")
        test_dir = os.path.join(self.data_dir, "test")
        
        print(f"Loaded training samples: {len(train_df)}")
        
        # Check for fixed validation set
        val_csv = os.path.join(self.data_dir, "val.csv")
        has_val_csv = os.path.exists(val_csv)
        
        if has_val_csv:
            # Use fixed validation set from val.csv
            val_df = pd.read_csv(val_csv)
            # Handle both 'StudyInstanceUID' and 'image' column names
            if 'StudyInstanceUID' in val_df.columns:
                val_images = set(val_df['StudyInstanceUID'].values)
            elif 'image' in val_df.columns:
                val_images = set(val_df['image'].values)
            else:
                raise ValueError("val.csv must have 'StudyInstanceUID' or 'image' column")
            
            # Remove validation samples from training data
            train_df = train_df[~train_df['StudyInstanceUID'].isin(val_images)]
            train_df = train_df.reset_index(drop=True)
            train_df['fold'] = 0  # No cross-validation
            folds = 1
            print(f"Using fixed validation set from val.csv: {len(val_df)} samples")
            print(f"Training samples after removing validation: {len(train_df)}")
        else:
            # Use GroupKFold for cross-validation based on PatientID
            gkf = GroupKFold(n_splits=self.folds)
            train_df['fold'] = -1
            for fold, (_, val_idx) in enumerate(gkf.split(train_df, groups=train_df['PatientID'])):
                train_df.loc[train_df.index[val_idx], 'fold'] = fold
            val_df = None
            folds = self.folds
            print(f"Using GroupKFold cross-validation with {folds} folds")
        
        # Define training transformations with data augmentation
        train_transform = T.Compose([
            T.Resize((self.img_size, self.img_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=10),
            T.ColorJitter(brightness=0.1, contrast=0.1),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Define validation/test transformations (no augmentation)
        val_transform = T.Compose([
            T.Resize((self.img_size, self.img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Load test data
        test_files = [f for f in os.listdir(test_dir) if f.endswith('.jpg')]
        test_df = pd.DataFrame({'StudyInstanceUID': [f.replace('.jpg', '') for f in test_files]})
        
        # Ensure test_df is in the same order as sample submission
        sample_submission_path = os.path.join(self.data_dir, "sample_submission.csv")
        if os.path.exists(sample_submission_path):
            sample_submission = pd.read_csv(sample_submission_path)
            test_df = test_df.merge(sample_submission[['StudyInstanceUID']], on='StudyInstanceUID', how='right')
        
        print(f"Test samples: {len(test_df)}")
        
        # Set train_data and test_data
        self.train_data = {
            'df': train_df,
            'val_df': val_df,
            'train_transform': train_transform,
            'val_transform': val_transform,
            'img_dir': train_dir,
            'target_cols': self.target_cols,
            'folds': folds,
            'has_val_csv': has_val_csv,
        }
        
        self.test_data = {
            'df': test_df,
            'transform': val_transform,
            'img_dir': test_dir,
            'target_cols': self.target_cols,
        }

    def describe(self) -> str:
        """Return a description of the data processing approach."""
        desc = f"""
        Chest X-Ray Data Loader for Catheter and Line Position Classification.
        
        Data Processing:
        - Loads training data from train.csv with {len(self.target_cols)} target columns
        - Uses fixed validation set from val.csv if available (strictly no random splitting)
        - Falls back to GroupKFold cross-validation based on PatientID for patient-wise splits
        - Removes validation samples from training data when val.csv exists
        
        Data Augmentation (training only):
        - Resize to {self.img_size}x{self.img_size}
        - Random horizontal flip (p=0.5)
        - Random rotation (±10 degrees)
        - Color jitter (brightness and contrast ±0.1)
        - ImageNet normalization
        
        Target Classes:
        {', '.join(self.target_cols)}
        """
        return desc