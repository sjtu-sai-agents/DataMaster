import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import warnings

warnings.filterwarnings("ignore")


def preprocess_metadata(df, is_train=True, encoders=None):
    """
    Preprocess metadata features.
    
    Args:
        df: DataFrame with metadata
        is_train: Whether this is training data
        encoders: Dictionary of encoders (for test data)
    
    Returns:
        df: Processed DataFrame
        metadata_tensor: Numpy array of metadata features
        encoders: Dictionary of encoders (for training data)
    """
    df = df.copy()
    
    # Fill missing values
    df["age_approx"].fillna(df["age_approx"].median(), inplace=True)
    df["sex"].fillna("unknown", inplace=True)
    df["anatom_site_general_challenge"].fillna("unknown", inplace=True)

    if is_train:
        # Encode categorical features
        le_sex = LabelEncoder()
        le_site = LabelEncoder()
        df["sex_encoded"] = le_sex.fit_transform(df["sex"])
        df["site_encoded"] = le_site.fit_transform(df["anatom_site_general_challenge"])
        
        # Standardize age
        age_scaler = StandardScaler()
        df["age_scaled"] = age_scaler.fit_transform(df[["age_approx"]])
        
        encoders = {
            "le_sex": le_sex,
            "le_site": le_site,
            "age_mean": age_scaler.mean_,
            "age_scale": age_scaler.scale_,
        }
    else:
        # Use provided encoders
        le_sex = encoders["le_sex"]
        le_site = encoders["le_site"]
        age_mean = encoders["age_mean"]
        age_scale = encoders["age_scale"]
        
        # Handle unseen labels by mapping to -1
        df["sex_encoded"] = df["sex"].apply(
            lambda x: le_sex.transform([x])[0] if x in le_sex.classes_ else -1
        )
        df["site_encoded"] = df["anatom_site_general_challenge"].apply(
            lambda x: le_site.transform([x])[0] if x in le_site.classes_ else -1
        )
        df["age_scaled"] = (df["age_approx"] - age_mean) / age_scale

    # Create metadata tensor
    meta_cols = ["age_scaled", "sex_encoded", "site_encoded"]
    metadata_tensor = df[meta_cols].values.astype(np.float32)

    return df, metadata_tensor, encoders


class MelanomaDataset(Dataset):
    """Dataset for melanoma classification."""
    
    def __init__(self, df, meta_tensor, img_dir, transform=None, is_train=True):
        self.df = df.reset_index(drop=True)
        self.meta_tensor = meta_tensor
        self.img_dir = img_dir
        self.transform = transform
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.loc[idx, "image_name"] + ".jpg"
        img_path = os.path.join(self.img_dir, img_name)

        # Load image
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Get metadata
        metadata = self.meta_tensor[idx]

        if self.is_train:
            target = self.df.loc[idx, "target"]
            return image, metadata, target
        else:
            return image, metadata


class MyDataLoader(BaseDataLoader):
    """Data loader for melanoma classification task."""
    
    def __init__(self, input_path="./input", batch_size=32, num_workers=4, 
                 img_size=256, **kwargs):
        super().__init__(**kwargs)
        self.input_path = input_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size
        self.encoders = None
        
    def setup(self):
        """Load data, preprocess, and create data loaders."""
        # Paths
        train_img_path = os.path.join(self.input_path, "jpeg/train")
        test_img_path = os.path.join(self.input_path, "jpeg/test")
        train_csv = os.path.join(self.input_path, "train.csv")
        test_csv = os.path.join(self.input_path, "test.csv")
        val_csv = os.path.join(self.input_path, "val.csv")
        
        # Load data
        train_df = pd.read_csv(train_csv)
        test_df = pd.read_csv(test_csv)
        
        # Preprocess training metadata
        train_df, train_meta, self.encoders = preprocess_metadata(train_df, is_train=True)
        
        # Save encoders for test set
        os.makedirs("./working", exist_ok=True)
        np.save("./working/le_sex_classes.npy", self.encoders["le_sex"].classes_)
        np.save("./working/le_site_classes.npy", self.encoders["le_site"].classes_)
        np.save("./working/age_scaler_mean.npy", self.encoders["age_mean"])
        np.save("./working/age_scaler_scale.npy", self.encoders["age_scale"])
        
        # Create validation split - use fixed val.csv if exists
        if os.path.exists(val_csv):
            # Use pre-defined validation set
            val_df_info = pd.read_csv(val_csv)
            # Check which column to use for image names
            if 'image' in val_df_info.columns:
                val_images = set(val_df_info['image'].values)
            elif 'image_name' in val_df_info.columns:
                val_images = set(val_df_info['image_name'].values)
            else:
                raise ValueError("val.csv must have 'image' or 'image_name' column")
            
            # Create masks for splitting
            val_mask = train_df['image_name'].isin(val_images).values
            train_mask = ~val_mask
            
            # Split data
            train_subset_df = train_df[train_mask].reset_index(drop=True)
            val_subset_df = train_df[val_mask].reset_index(drop=True)
            train_meta_subset = train_meta[train_mask]
            val_meta_subset = train_meta[val_mask]
        else:
            # Fallback to stratified random split only if val.csv doesn't exist
            train_idx, val_idx = train_test_split(
                np.arange(len(train_df)),
                test_size=0.2,
                random_state=42,
                stratify=train_df["target"],
            )
            train_subset_df = train_df.iloc[train_idx].reset_index(drop=True)
            val_subset_df = train_df.iloc[val_idx].reset_index(drop=True)
            train_meta_subset = train_meta[train_idx]
            val_meta_subset = train_meta[val_idx]
        
        # Define transforms
        train_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
        # Create datasets
        train_dataset = MelanomaDataset(
            train_subset_df, train_meta_subset, train_img_path, 
            train_transform, is_train=True
        )
        val_dataset = MelanomaDataset(
            val_subset_df, val_meta_subset, train_img_path, 
            val_transform, is_train=True
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, 
            num_workers=self.num_workers, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False, 
            num_workers=self.num_workers, pin_memory=True
        )
        
        # Prepare test data
        test_df, test_meta, _ = preprocess_metadata(test_df, is_train=False, encoders=self.encoders)
        test_dataset = MelanomaDataset(
            test_df, test_meta, test_img_path, val_transform, is_train=False
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False, 
            num_workers=self.num_workers, pin_memory=True
        )
        
        # Set train_data and test_data
        self.train_data = {
            'train_loader': train_loader,
            'val_loader': val_loader,
            'val_df': val_subset_df,
        }
        self.test_data = {
            'test_loader': test_loader,
            'test_df': test_df,
        }
        
    def describe(self) -> str:
        """Return description of data processing approach."""
        return """
        Melanoma Classification Data Loader
        
        Data Processing:
        - Loads JPEG images from input/jpeg/train and input/jpeg/test directories
        - Processes metadata features: age_approx, sex, anatom_site_general_challenge
        - Fills missing values with median/unknown
        - Encodes categorical features with LabelEncoder
        - Standardizes age feature with StandardScaler
        
        Data Augmentation (Training):
        - Resize to 256x256
        - Random horizontal flip
        - Random vertical flip
        - Random rotation (20 degrees)
        - Normalization with ImageNet statistics
        
        Validation Split:
        - Uses input/val.csv if available (fixed validation set)
        - Falls back to stratified random split if val.csv not found
        """