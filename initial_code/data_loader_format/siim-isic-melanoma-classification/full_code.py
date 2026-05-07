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

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Melanoma Classification Training')
    
    # Model hyperparameters
    parser.add_argument('--meta_dim', type=int, default=3,
                        help='Dimension of metadata features')
    parser.add_argument('--img_size', type=int, default=256,
                        help='Image size for resizing')
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay for optimizer')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Scheduler parameters
    parser.add_argument('--scheduler_patience', type=int, default=2,
                        help='Patience for learning rate scheduler')
    parser.add_argument('--scheduler_factor', type=float, default=0.5,
                        help='Factor for learning rate scheduler')
    
    # Path parameters
    parser.add_argument('--input_path', type=str, default='./input',
                        help='Path to input data directory')
    parser.add_argument('--working_path', type=str, default='./working',
                        help='Path to working directory for saving models')
    parser.add_argument('--submission_path', type=str, default='./submission',
                        help='Path to save submission file')
    
    return parser.parse_args()


class MelanomaModel(nn.Module):
    """Melanoma classification model combining image and metadata features."""
    
    def __init__(self, meta_dim=3):
        super(MelanomaModel, self).__init__()
        # Image branch (ResNet50)
        self.img_model = models.resnet50(pretrained=True)
        self.img_model.fc = nn.Identity()
        img_features = 2048

        # Metadata branch
        self.meta_fc = nn.Sequential(
            nn.Linear(meta_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        meta_features = 32

        # Combined classifier
        self.classifier = nn.Sequential(
            nn.Linear(img_features + meta_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img, meta):
        # Image features
        img_features = self.img_model(img)

        # Metadata features
        meta_features = self.meta_fc(meta)

        # Concatenate and classify
        combined = torch.cat([img_features, meta_features], dim=1)
        output = self.classifier(combined)
        return output.squeeze()


def main():
    """Main training function."""
    args = parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs(args.working_path, exist_ok=True)
    os.makedirs(args.submission_path, exist_ok=True)
    
    # Initialize data loader and get data
    data_loader = MyDataLoader(
        input_path=args.input_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size
    )
    train_data, test_data = data_loader.get_data()
    
    train_loader = train_data['train_loader']
    val_loader = train_data['val_loader']
    val_df = train_data['val_df']
    test_loader = test_data['test_loader']
    test_df = test_data['test_df']
    
    # Initialize model
    model = MelanomaModel(meta_dim=args.meta_dim).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=args.scheduler_patience, factor=args.scheduler_factor
    )
    
    # Training loop
    best_val_auc = 0
    
    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss = 0
        train_preds = []
        train_targets = []

        for images, metadata, targets in train_loader:
            images = images.to(device)
            metadata = metadata.to(device)
            targets = targets.float().to(device)

            optimizer.zero_grad()
            outputs = model(images, metadata)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_preds.extend(outputs.detach().cpu().numpy())
            train_targets.extend(targets.cpu().numpy())

        # Validation
        model.eval()
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for images, metadata, targets in val_loader:
                images = images.to(device)
                metadata = metadata.to(device)
                targets = targets.float().to(device)

                outputs = model(images, metadata)
                val_preds.extend(outputs.cpu().numpy())
                val_targets.extend(targets.cpu().numpy())

        # Calculate metrics
        train_auc = roc_auc_score(train_targets, train_preds)
        val_auc = roc_auc_score(val_targets, val_preds)
        avg_train_loss = train_loss / len(train_loader)

        print(f"Epoch {epoch+1}/{args.epochs}:")
        print(f"Train Loss: {avg_train_loss:.4f}, Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}")

        # Update scheduler
        scheduler.step(val_auc)

        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), os.path.join(args.working_path, "best_model.pth"))

    print(f"\nBest Validation AUC: {best_val_auc:.4f}")
    
    # Load best model and predict on test set
    model.load_state_dict(torch.load(os.path.join(args.working_path, "best_model.pth")))
    model.eval()

    test_preds = []

    with torch.no_grad():
        for images, metadata in test_loader:
            images = images.to(device)
            metadata = metadata.to(device)

            outputs = model(images, metadata)
            test_preds.extend(outputs.cpu().numpy())

    # Create submission file
    submission_df = pd.DataFrame({
        "image_name": test_df["image_name"],
        "target": test_preds
    })
    
    submission_file = os.path.join(args.submission_path, "submission.csv")
    submission_df.to_csv(submission_file, index=False)

    print(f"\nSubmission saved to {submission_file}")
    print(f"Submission shape: {submission_df.shape}")
    print(f"Target range: [{submission_df['target'].min():.3f}, {submission_df['target'].max():.3f}]")

    # Save validation predictions for reference
    val_df_copy = val_df.copy()
    val_df_copy["prediction"] = val_preds
    val_df_copy[["image_name", "target", "prediction"]].to_csv(
        os.path.join(args.working_path, "validation_predictions.csv"), index=False
    )
    print(f"\nValidation AUC: {best_val_auc:.4f}")


if __name__ == "__main__":
    main()