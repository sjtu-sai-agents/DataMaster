import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight


class ImageFeatureExtractor:
    """Extract image features using pre-trained EfficientNet-B0."""
    
    def __init__(self, device):
        self.device = device
        self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.model.classifier = nn.Identity()
        self.model = self.model.to(device)
        self.model.eval()
        self.preprocess = models.EfficientNet_B0_Weights.DEFAULT.transforms()
    
    def extract(self, ids, image_dir):
        """Extract features for a list of image IDs."""
        features = []
        for id_val in ids:
            img_path = os.path.join(image_dir, f"{id_val}.jpg")
            img = Image.open(img_path).convert('RGB')
            img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feat = self.model(img_tensor)
            features.append(feat.cpu().numpy().flatten())
        return np.array(features, dtype=np.float32)


class MyDataLoader(BaseDataLoader):
    """Data loader for leaf classification task."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def setup(self):
        """
        Load data, preprocess features, extract image features, and prepare train/val/test splits.
        Uses fixed validation set from input/val.csv if available.
        """
        # Load data
        train_df = pd.read_csv('./input/train.csv')
        test_df = pd.read_csv('./input/test.csv')
        
        # Tabular features: all columns except id and species
        feature_cols = [col for col in train_df.columns if col not in ['id', 'species']]
        X_train_tab = train_df[feature_cols].values.astype(np.float32)
        X_test_tab = test_df[feature_cols].values.astype(np.float32)
        
        # Scale features
        scaler = StandardScaler()
        X_train_tab_scaled = scaler.fit_transform(X_train_tab)
        X_test_tab_scaled = scaler.transform(X_test_tab)
        
        # Encode labels
        le = LabelEncoder()
        y_train = le.fit_transform(train_df['species'])
        num_classes = len(le.classes_)
        class_names = le.classes_
        
        # Class weights (for loss) - computed from full training set
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        
        # Extract image features
        image_dir = './input/images'
        img_extractor = ImageFeatureExtractor(self.device)
        
        print("Extracting image features for training set...")
        train_img_feats = img_extractor.extract(train_df['id'].values, image_dir)
        print("Extracting image features for test set...")
        test_img_feats = img_extractor.extract(test_df['id'].values, image_dir)
        
        # Combine tabular and image features
        X_train = np.hstack([X_train_tab_scaled, train_img_feats])
        X_test = np.hstack([X_test_tab_scaled, test_img_feats])
        
        # Handle validation split - MUST use fixed val.csv if available
        if os.path.exists('./input/val.csv'):
            val_df = pd.read_csv('./input/val.csv')
            val_ids = set(val_df['id'].values)
            
            # Create masks for train and validation
            train_mask = np.array([id_val not in val_ids for id_val in train_df['id'].values])
            val_mask = np.array([id_val in val_ids for id_val in train_df['id'].values])
            
            X_train_final = X_train[train_mask]
            y_train_final = y_train[train_mask]
            X_val = X_train[val_mask]
            y_val = y_train[val_mask]
            
            print(f"Using fixed validation set from val.csv: {len(X_val)} samples")
        else:
            # Fallback: split from original competition data only
            from sklearn.model_selection import train_test_split
            X_train_final, X_val, y_train_final, y_val = train_test_split(
                X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
            )
            print("Warning: val.csv not found, using train_test_split for validation")
        
        # Store processed data
        self.train_data = {
            'X_train': X_train_final,
            'y_train': y_train_final,
            'X_val': X_val,
            'y_val': y_val,
            'class_weights': class_weights,
            'num_classes': num_classes,
            'class_names': class_names
        }
        
        self.test_data = {
            'X_test': X_test,
            'test_ids': test_df['id'].values,
            'class_names': class_names
        }
    
    def describe(self):
        """
        Return a description of the data processing approach.
        """
        return ("Data loader for leaf classification. "
                "Extracts tabular features with StandardScaler normalization "
                "and image features using pre-trained EfficientNet-B0. "
                "Uses fixed validation set from input/val.csv if available, "
                "otherwise falls back to stratified train_test_split.")