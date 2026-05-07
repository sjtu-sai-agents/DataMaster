import os
import sys
import random
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms as transforms
import timm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tqdm import tqdm
from abc import ABC, abstractmethod


# ----------------------------------------------------------------------
# BaseDataLoader definition (DO NOT MODIFY)
# ----------------------------------------------------------------------
class BaseDataLoader(ABC):
    """
    Abstract base class for data loaders.

    This class defines the interface that all data loaders must implement.
    Subclasses only need to implement:
    - setup(): Load and process data
    - describe(): Provide a description

    Data splitting is NOT handled here - it's the responsibility of the training code.

    Attributes:
        config: Additional keyword arguments for customization
        train_data: Processed training data (set by setup())
        test_data: Processed test data (set by setup())
    """

    def __init__(self, **kwargs):
        """
        Initialize the BaseDataLoader.

        Args:
            **kwargs: Additional configuration (accessible via self.config)
        """
        self.config = kwargs
        self.train_data = None
        self.test_data = None
        self._is_setup = False

    # ========================================================================
    # Abstract methods - must be implemented by subclasses
    # ========================================================================

    @abstractmethod
    def setup(self):
        """
        Setup the data loader.

        Subclasses must implement this method to:
        - Load data from disk (CSV, images, etc.)
        - Perform feature engineering
        - Define data augmentation strategies
        - Set self.train_data and self.test_data

        Note: Do NOT perform train/val split here. Splitting is handled by the training code.
        """
        raise NotImplementedError(
            "Methods should implement `setup` methods for specific data loader class!"
        )

    @abstractmethod
    def describe(self) -> str:
        """
        Return a description of this data loader.

        Should describe:
        - What task this loader is for
        - What data processing/augmentation tricks is applied
        - What new data source and datsets are added.
        - Any notable features of the implementation

        Returns:
            A string description of the data loader
        """
        raise NotImplementedError(
            "Methods should implement `describe` methods for specific data loader class!"
        )

    def get_data(self):
        """
        Return the processed training and test data.

        This is the main interface - returns the data after all
        preprocessing, feature engineering, and augmentation.

        Automatically calls setup() if not already called.
        """
        if not self._is_setup:
            print(
                "Warning: Using Base Methods get_data: load self.train_data & self.test_data"
            )
            self.setup()
            self._is_setup = True

        return self.train_data, self.test_data

    def __str__(self) -> str:
        return self.describe()


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 8
LR = 2e-4
WEIGHT_DECAY = 1e-4
DROP_PATH_RATE = 0.1
TRAIN_SPLIT = 0.85
VAL_SPLIT = 0.15
NUM_WORKERS = max(4, min(8, os.cpu_count() or 4))
MAX_TRAIN_SAMPLES = 2000
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ----------------------------------------------------------------------
# Set seeds and CUDA settings (non‑deterministic for speed)
# ----------------------------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

set_seed(SEED)


# ----------------------------------------------------------------------
# Image Dataset class
# ----------------------------------------------------------------------
class ImageDataset(Dataset):
    def __init__(self, root_dir, image_list, label_list=None, transform=None):
        self.root_dir = root_dir
        self.image_list = image_list
        self.label_list = label_list
        self.has_labels = label_list is not None
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_list[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        if self.has_labels:
            label = torch.tensor(self.label_list[idx], dtype=torch.float)
            return image, label, self.image_list[idx]
        else:
            return image, self.image_list[idx]


# ----------------------------------------------------------------------
# DataLoader Implementation
# ----------------------------------------------------------------------
class MyDataLoader(BaseDataLoader):
    """
    DataLoader for Plant Pathology 2021 - FGVC8 competition.
    Loads apple leaf images and prepares them for multi-label disease classification.
    """
    
    def setup(self):
        """Load and process the apple foliar disease dataset."""
        # Load CSV files
        train_full_df = pd.read_csv('input/train.csv')

        # ============================================
        # ✅ Use pre-split validation set when available
        # ============================================
        has_presplit_val = os.path.exists('input/val.csv')
        if has_presplit_val:
            val_df = pd.read_csv('input/val.csv')

            # Remove validation samples from training set so train/val stay disjoint.
            val_images = set(val_df['image'].values)
            train_df = train_full_df[~train_full_df['image'].isin(val_images)].reset_index(drop=True)

            print(f'✅ Using pre-split validation set: {len(val_df)} samples')
            print(f'   Training set (after removing val): {len(train_df)} samples')
        else:
            val_df = None
            train_df = train_full_df
            print('⚠️  WARNING: input/val.csv not found, falling back to random split')

        # Sample to reduce training time (~2min target, full set ~15k)
        if len(train_df) > MAX_TRAIN_SAMPLES:
            train_df = train_df.sample(n=MAX_TRAIN_SAMPLES, random_state=SEED, replace=False)
            train_df = train_df.reset_index(drop=True)

        # Extract disease classes from both train and val so the label space stays fixed.
        all_labels = set()
        for s in train_df['labels']:
            all_labels.update(s.split())
        if val_df is not None:
            for s in val_df['labels']:
                all_labels.update(s.split())
        classes = sorted(list(all_labels))
        num_classes = len(classes)
        class_to_idx = {c: i for i, c in enumerate(classes)}
        idx_to_class = {i: c for i, c in enumerate(classes)}

        def encode_labels(df):
            encoded = np.zeros((len(df), num_classes), dtype=np.float32)
            for i, s in enumerate(df['labels']):
                for disease in s.split():
                    encoded[i, class_to_idx[disease]] = 1.0
            return encoded

        train_binary_labels = encode_labels(train_df)

        if has_presplit_val:
            train_images = train_df['image'].values
            train_labels = train_binary_labels
            val_images = val_df['image'].values
            val_labels = encode_labels(val_df)
        else:
            indices = np.arange(len(train_df))
            train_idx, val_idx = train_test_split(
                indices, test_size=VAL_SPLIT, random_state=SEED, shuffle=True
            )

            train_images = train_df['image'].values[train_idx]
            val_images = train_df['image'].values[val_idx]
            train_labels = train_binary_labels[train_idx]
            val_labels = train_binary_labels[val_idx]

        # IMPORTANT: Get test images from actual directory, NOT from sample_submission.csv
        # The sample_submission.csv may have different image names than the actual test set
        test_image_dir = 'input/test_images'
        test_images = sorted([f for f in os.listdir(test_image_dir) if f.endswith(('.jpg', '.png'))])

        # Image transforms
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(20, fill=(128,128,128)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])

        eval_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])

        # Create Dataset instances
        train_dataset = ImageDataset(
            root_dir='input/train_images',
            image_list=train_images,
            label_list=train_labels,
            transform=train_transform
        )

        val_dataset = ImageDataset(
            root_dir='input/train_images',
            image_list=val_images,
            label_list=val_labels,
            transform=eval_transform
        )

        test_dataset = ImageDataset(
            root_dir='input/test_images',
            image_list=test_images,
            label_list=None,
            transform=eval_transform
        )

        # Create DataLoader instances
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            drop_last=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            drop_last=False
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            drop_last=False
        )

        # Compute class weights for loss
        pos_counts = train_labels.sum(axis=0)
        neg_counts = len(train_labels) - pos_counts
        pos_weight = neg_counts / (pos_counts + 1e-7)  # avoid division by zero
        pos_weight = torch.tensor(pos_weight, dtype=torch.float, device=device)

        # Store all data in train_data (as a dictionary for multiple return values)
        self.train_data = {
            'train_loader': train_loader,
            'val_loader': val_loader,
            'test_loader': test_loader,
            'classes': classes,
            'class_to_idx': class_to_idx,
            'idx_to_class': idx_to_class,
            'num_classes': num_classes,
            'pos_weight': pos_weight,
            'train_dataset': train_dataset,
            'val_dataset': val_dataset,
            'test_dataset': test_dataset,
            'test_images': test_images
        }

    def describe(self) -> str:
        """Return description of the data loader."""
        return (
            "Plant Pathology 2021 DataLoader for apple foliar disease classification. "
            "Loads RGB images from train_images and test_images directories. "
            "Performs multi-label encoding for disease categories. "
            "Uses random augmentations (resize, flips, color jitter, rotation) for training. "
            "Computes class weights for imbalanced loss. "
            "Returns train/val/test DataLoaders with ImageNet normalization."
        )

    def get_data(self):
        """Return all processed data as a dictionary."""
        if not self._is_setup:
            self.setup()
            self._is_setup = True
        return self.train_data


# ----------------------------------------------------------------------
# TTA prediction function
# ----------------------------------------------------------------------
def predict_tta(model, loader, device, has_labels=True):
    model.eval()
    all_probs = []
    all_targets = [] if has_labels else None
    with torch.no_grad():
        for batch in tqdm(loader, desc='TTA', leave=False):
            if has_labels:
                images, labels, _ = batch
                if all_targets is not None:
                    all_targets.append(labels.cpu())
            else:
                images, _ = batch
            images = images.to(device)

            # Original
            logits = model(images)
            probs = torch.sigmoid(logits)

            # Horizontal flip
            images_h = torch.flip(images, dims=[3])
            logits_h = model(images_h)
            probs_h = torch.sigmoid(logits_h)

            # Vertical flip
            images_v = torch.flip(images, dims=[2])
            logits_v = model(images_v)
            probs_v = torch.sigmoid(logits_v)

            # Both flips
            images_both = torch.flip(images, dims=[2,3])
            logits_both = model(images_both)
            probs_both = torch.sigmoid(logits_both)

            avg_probs = (probs + probs_h + probs_v + probs_both) / 4.0
            all_probs.append(avg_probs.cpu())
    all_probs = torch.cat(all_probs, dim=0).numpy()
    if has_labels:
        all_targets = torch.cat(all_targets, dim=0).numpy()
        return all_probs, all_targets
    else:
        return all_probs


# ----------------------------------------------------------------------
# Main algorithm
# ----------------------------------------------------------------------
def main():
    # Initialize DataLoader
    loader = MyDataLoader()
    data = loader.get_data()

    # Extract data from loader
    train_loader = data['train_loader']
    val_loader = data['val_loader']
    test_loader = data['test_loader']
    classes = data['classes']
    class_to_idx = data['class_to_idx']
    idx_to_class = data['idx_to_class']
    num_classes = data['num_classes']
    pos_weight = data['pos_weight']
    train_dataset = data['train_dataset']
    val_dataset = data['val_dataset']
    test_images = data['test_images']

    # ----------------------------------------------------------------------
    # Model
    # ----------------------------------------------------------------------
    model = timm.create_model(
        'vit_base_patch16_224',
        pretrained=True,
        num_classes=num_classes,
        drop_path_rate=DROP_PATH_RATE
    )
    model = model.to(device)

    # ----------------------------------------------------------------------
    # Loss, Optimizer, Scheduler
    # ----------------------------------------------------------------------
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    scaler = GradScaler()
    best_f1 = 0.0
    best_model_path = 'best_model.pth'

    # ----------------------------------------------------------------------
    # Training and validation (without TTA)
    # ----------------------------------------------------------------------
    print(f'Training on {len(train_dataset)} images, validating on {len(val_dataset)} images.')
    for epoch in range(1, EPOCHS+1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{EPOCHS}', leave=False)
        for images, labels, _ in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * images.size(0)
            pbar.set_postfix({'loss': loss.item()})
        scheduler.step()
        epoch_loss = running_loss / len(train_dataset)

        # Validation
        model.eval()
        val_outputs, val_targets = [], []
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images = images.to(device)
                outputs = model(images)
                val_outputs.append(outputs.cpu())
                val_targets.append(labels.cpu())
        val_outputs = torch.cat(val_outputs, dim=0)
        val_probs = torch.sigmoid(val_outputs).numpy()
        val_targets = torch.cat(val_targets, dim=0).numpy()
        preds = (val_probs > 0.5).astype(int)
        f1 = f1_score(val_targets, preds, average='macro')
        print(f'Epoch {epoch:2d} | Loss: {epoch_loss:.4f} | Val F1 (0.5): {f1:.4f}')

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), best_model_path)
            print(f'  -> New best model saved (F1={f1:.4f})')

    # ----------------------------------------------------------------------
    # Load best model
    # ----------------------------------------------------------------------
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()
    print('Best model loaded.')

    # ----------------------------------------------------------------------
    # Obtain TTA probabilities on validation set and optimize thresholds
    # ----------------------------------------------------------------------
    val_probs_tta, val_targets_tta = predict_tta(model, val_loader, device, has_labels=True)

    opt_thresholds = np.zeros(num_classes)
    print('Optimizing per-class thresholds...')
    for i in range(num_classes):
        best_thresh = 0.5
        best_f1 = 0
        for thresh in np.arange(0.05, 1.0, 0.05):
            pred = (val_probs_tta[:, i] >= thresh).astype(int)
            f1 = f1_score(val_targets_tta[:, i], pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        opt_thresholds[i] = best_thresh
        print(f'  class {classes[i]:20s} -> threshold {best_thresh:.2f} (F1={best_f1:.3f})')

    # ----------------------------------------------------------------------
    # Compute final validation predictions with optimized thresholds
    # ----------------------------------------------------------------------
    val_preds = np.zeros_like(val_probs_tta)
    for i in range(num_classes):
        val_preds[:, i] = (val_probs_tta[:, i] >= opt_thresholds[i]).astype(int)

    # Ensure at least one prediction per sample
    empty_rows = np.sum(val_preds, axis=1) == 0
    if np.any(empty_rows):
        max_indices = np.argmax(val_probs_tta[empty_rows], axis=1)
        for j, idx in enumerate(np.where(empty_rows)[0]):
            val_preds[idx, max_indices[j]] = 1

    final_val_f1 = f1_score(val_targets_tta, val_preds, average='macro')
    print(f'\nValidation Macro F1 (optimized thresholds): {final_val_f1:.5f}\n')

    # ----------------------------------------------------------------------
    # Predict on test set and create submission
    # ----------------------------------------------------------------------
    test_probs_tta = predict_tta(model, test_loader, device, has_labels=False)

    test_preds = np.zeros_like(test_probs_tta)
    for i in range(num_classes):
        test_preds[:, i] = (test_probs_tta[:, i] >= opt_thresholds[i]).astype(int)

    empty_rows = np.sum(test_preds, axis=1) == 0
    if np.any(empty_rows):
        max_indices = np.argmax(test_probs_tta[empty_rows], axis=1)
        for j, idx in enumerate(np.where(empty_rows)[0]):
            test_preds[idx, max_indices[j]] = 1

    # Convert to space‑separated label strings
    test_labels_str = []
    for i in range(len(test_preds)):
        indices = np.where(test_preds[i])[0]
        labels = [idx_to_class[idx] for idx in indices]
        test_labels_str.append(' '.join(labels))

    submission = pd.DataFrame({
        'image': test_images,
        'labels': test_labels_str
    })

    os.makedirs('submission', exist_ok=True)
    submission.to_csv('submission/submission.csv', index=False)
    print('Submission saved to submission/submission.csv')


if __name__ == "__main__":
    main()
