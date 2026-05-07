import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


class MyDataLoader(BaseDataLoader):
    """
    DataLoader for image classification task.
    Handles data loading, train/val split, and test data preparation.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_dir = kwargs.get('input_dir', './input')
        self.train_dir = os.path.join(self.input_dir, 'train')
        self.test_dir = os.path.join(self.input_dir, 'test')
        self.img_size = kwargs.get('img_size', 224)
        self.val_split = kwargs.get('val_split', 0.2)
        self.seed = kwargs.get('seed', 42)
        
    def setup(self):
        """Load data, handle train/val split, prepare test data."""
        # Load training data
        train_files, train_labels = self._load_train_data()
        print(f"Found {len(train_files)} training images.")
        
        # Load test data
        test_files, test_ids = self._load_test_data()
        print(f"Found {len(test_files)} test images.")
        
        # Handle train/val split - MUST use val.csv if it exists
        tr_files, tr_labels, va_files, va_labels = self._split_train_val(train_files, train_labels)
        print(f"Train: {len(tr_files)}, Validation: {len(va_files)}")
        
        # Set train_data and test_data
        self.train_data = {
            'train_files': tr_files,
            'train_labels': tr_labels,
            'val_files': va_files,
            'val_labels': va_labels,
            'img_size': self.img_size
        }
        self.test_data = {
            'test_files': test_files,
            'test_ids': test_ids,
            'img_size': self.img_size
        }
    
    def _load_train_data(self):
        """Load training files and labels from train directory."""
        train_files, train_labels = [], []
        for fname in os.listdir(self.train_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                label = 1 if fname.lower().startswith('dog') else 0
                train_files.append(os.path.join(self.train_dir, fname))
                train_labels.append(label)
        return np.array(train_files), np.array(train_labels)
    
    def _load_test_data(self):
        """Load test files and IDs from sample_submission.csv or test directory."""
        sample_sub_path = os.path.join(self.input_dir, 'sample_submission.csv')
        if os.path.exists(sample_sub_path):
            sample_df = pd.read_csv(sample_sub_path)
            test_ids = sample_df['id'].values
            test_files = np.array([os.path.join(self.test_dir, f"{i}.jpg") for i in test_ids])
            # Verify existence
            missing = [f for f in test_files if not os.path.exists(f)]
            if missing:
                print(f"Warning: {len(missing)} test files missing, fallback to scanning directory.")
                return self._scan_test_directory()
            return test_files, test_ids
        else:
            print("sample_submission.csv not found, scanning test directory.")
            return self._scan_test_directory()
    
    def _scan_test_directory(self):
        """Scan test directory for image files."""
        test_files, test_ids = [], []
        for fname in os.listdir(self.test_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                try:
                    iid = int(os.path.splitext(fname)[0])
                except:
                    continue
                test_ids.append(iid)
                test_files.append(os.path.join(self.test_dir, fname))
        sort_idx = np.argsort(test_ids)
        test_ids = np.array(test_ids)[sort_idx]
        test_files = np.array(test_files)[sort_idx]
        return test_files, test_ids
    
    def _split_train_val(self, train_files, train_labels):
        """
        Split training data into train and validation sets.
        MUST use val.csv if it exists - random splitting is strictly prohibited when val.csv exists.
        """
        val_csv_path = os.path.join(self.input_dir, 'val.csv')
        if os.path.exists(val_csv_path):
            return self._split_by_val_csv(train_files, train_labels, val_csv_path)
        else:
            return self._stratified_split(train_files, train_labels)
    
    def _split_by_val_csv(self, train_files, train_labels, val_csv_path):
        """Split using val.csv file - uses fixed validation set."""
        val_df = pd.read_csv(val_csv_path)
        
        # Determine column name for images
        if 'image' in val_df.columns:
            val_images = set(val_df['image'].values)
        elif 'filename' in val_df.columns:
            val_images = set(val_df['filename'].values)
        else:
            # Assume first column contains image names
            val_images = set(val_df.iloc[:, 0].values)
        
        # Create masks - check both full path and basename
        is_val = np.array([
            os.path.basename(f) in val_images or f in val_images 
            for f in train_files
        ])
        
        if np.sum(is_val) == 0:
            print("Warning: No validation files found in train directory. Falling back to stratified split.")
            return self._stratified_split(train_files, train_labels)
        
        tr_files = train_files[~is_val]
        tr_labels = train_labels[~is_val]
        va_files = train_files[is_val]
        va_labels = train_labels[is_val]
        
        return tr_files, tr_labels, va_files, va_labels
    
    def _stratified_split(self, train_files, train_labels):
        """Split using stratified shuffle split (only when val.csv doesn't exist)."""
        sss = StratifiedShuffleSplit(n_splits=1, test_size=self.val_split, random_state=self.seed)
        train_idx, val_idx = next(sss.split(train_files, train_labels))
        return train_files[train_idx], train_labels[train_idx], train_files[val_idx], train_labels[val_idx]
    
    def describe(self):
        """Return description of the data loader."""
        return """DataLoader for image classification task (cat vs dog).
        
Features:
- Loads training images from train directory (classification based on filename prefix: 'dog' = 1, 'cat' = 0)
- Loads test images from test directory (ordered by sample_submission.csv or filename)
- Train/val split: Uses fixed val.csv if available, otherwise stratified 80/20 split
- Supports common image formats: jpg, jpeg, png, bmp
- Returns file paths and labels for train/val/test sets
"""