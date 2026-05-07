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