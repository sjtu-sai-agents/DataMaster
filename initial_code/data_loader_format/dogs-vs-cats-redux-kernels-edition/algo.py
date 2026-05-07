import os
import random
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import timm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from contextlib import nullcontext


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ImagePathDataset(Dataset):
    """Custom Dataset for loading images from file paths."""
    
    def __init__(self, files, labels=None, transform=None):
        self.files = files
        self.labels = labels if labels is not None else None
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        if self.labels is not None:
            return img, int(self.labels[idx])
        else:
            return img, -1


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Image Classification Training Script')
    
    # Path arguments
    parser.add_argument('--input_dir', type=str, default='./input',
                        help='Input directory containing train and test folders')
    parser.add_argument('--submission_dir', type=str, default='./submission',
                        help='Directory to save submission files')
    parser.add_argument('--working_dir', type=str, default='./working',
                        help='Working directory for intermediate files')
    
    # Model arguments
    parser.add_argument('--img_size', type=int, default=224,
                        help='Image size for model input')
    parser.add_argument('--model_names', type=str, nargs='+',
                        default=['vit_large_patch14_224.clip', 'swin_large_patch4_window7_224',
                                 'convnext_large.fb_in22k_ft_in1k', 'vit_base_patch16_224'],
                        help='List of model names to try loading (in order)')
    
    # Training arguments
    parser.add_argument('--batch_size_train', type=int, default=128,
                        help='Batch size for training feature extraction')
    parser.add_argument('--batch_size_tta', type=int, default=64,
                        help='Batch size for TTA feature extraction')
    parser.add_argument('--C_candidates', type=float, nargs='+',
                        default=[0.25, 0.5, 1.0, 2.0, 4.0],
                        help='C values for Logistic Regression hyperparameter tuning')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation split ratio (used only if val.csv not found)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    return parser.parse_args()


def get_num_workers():
    """Get optimal number of workers for DataLoader."""
    cpu_count = os.cpu_count()
    if cpu_count is None:
        return 4
    else:
        return max(4, min(12, cpu_count - 1 if cpu_count > 1 else 4))


def make_loader(dataset, batch_size, shuffle=False, num_workers=4):
    """Create DataLoader with standard settings."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        drop_last=False
    )


def extract_features(model, loader, device, is_tta=False, chunk_size=None):
    """Extract features from images using the model."""
    model.eval()
    features = []
    autocast = torch.cuda.amp.autocast if torch.cuda.is_available() else nullcontext
    
    with torch.no_grad():
        for imgs, _ in loader:
            if is_tta:
                B, N_crops, C, H, W = imgs.shape
                imgs = imgs.view(B * N_crops, C, H, W).to(device, non_blocking=True)
                out_list = []
                for i in range(0, imgs.size(0), chunk_size):
                    chunk = imgs[i:i+chunk_size]
                    with autocast():
                        feat = model(chunk)
                    out_list.append(feat.cpu())
                feats = torch.cat(out_list, dim=0)
                feats = feats.view(B, N_crops, -1).mean(dim=1)  # Average over crops
            else:
                imgs = imgs.to(device, non_blocking=True)
                out_list = []
                for i in range(0, imgs.size(0), chunk_size):
                    chunk = imgs[i:i+chunk_size]
                    with autocast():
                        feat = model(chunk)
                    out_list.append(feat.cpu())
                feats = torch.cat(out_list, dim=0)
            features.append(feats.numpy())
    
    return np.vstack(features)


def l2_normalize(X):
    """L2 normalize feature vectors."""
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    norm = np.maximum(norm, 1e-12)
    return X / norm


def main():
    """Main training function."""
    args = parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Create directories
    os.makedirs(args.submission_dir, exist_ok=True)
    os.makedirs(args.working_dir, exist_ok=True)
    
    # Load data using MyDataLoader
    data_loader = MyDataLoader(
        input_dir=args.input_dir,
        img_size=args.img_size,
        val_split=args.val_split,
        seed=args.seed
    )
    train_data, test_data = data_loader.get_data()
    
    # Extract data from loader
    tr_files = train_data['train_files']
    tr_labels = train_data['train_labels']
    va_files = train_data['val_files']
    va_labels = train_data['val_labels']
    test_files = test_data['test_files']
    test_ids = test_data['test_ids']
    img_size = train_data['img_size']
    
    # Define image transforms
    IMG_MEAN = [0.485, 0.456, 0.406]
    IMG_STD = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=IMG_MEAN, std=IMG_STD)
    
    # Base transform (no augmentation)
    base_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        normalize
    ])
    
    # Horizontal flip transform (augmentation)
    hflip_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        normalize
    ])
    
    # TenCrop transform for Test-Time Augmentation (TTA)
    tencrop_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.TenCrop(img_size),
        transforms.Lambda(lambda crops: torch.stack(
            [normalize(transforms.ToTensor()(crop)) for crop in crops]
        ))
    ])
    
    # Create datasets
    train_ds_base = ImagePathDataset(tr_files, tr_labels, transform=base_transform)
    train_ds_flip = ImagePathDataset(tr_files, tr_labels, transform=hflip_transform)
    val_ds_tta = ImagePathDataset(va_files, va_labels, transform=tencrop_transform)
    test_ds_tta = ImagePathDataset(test_files, None, transform=tencrop_transform)
    
    # Create dataloaders
    num_workers = get_num_workers()
    print(f"DataLoader using {num_workers} workers")
    
    train_loader_base = make_loader(train_ds_base, args.batch_size_train, num_workers=num_workers)
    train_loader_flip = make_loader(train_ds_flip, args.batch_size_train, num_workers=num_workers)
    val_loader_tta = make_loader(val_ds_tta, args.batch_size_tta, num_workers=num_workers)
    test_loader_tta = make_loader(test_ds_tta, args.batch_size_tta, num_workers=num_workers)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load pretrained model from timm
    model = None
    selected_model_name = None
    for name in args.model_names:
        try:
            print(f"Trying to load {name}...")
            model = timm.create_model(name, pretrained=True, num_classes=0)
            print(f"Loaded {name} successfully.")
            selected_model_name = name
            break
        except Exception as e:
            print(f"Failed to load {name}: {e}")
    
    if model is None:
        print("All models failed, falling back to vit_base_patch16_224 without pretrained weights.")
        model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0)
        selected_model_name = "vit_base_patch16_224"
    
    model = model.to(device)
    model.eval()
    print(f"Model: {selected_model_name}")
    
    # Determine chunk sizes for feature extraction (avoid OOM)
    large_models = ["vit_large_patch14_224.clip", "swin_large_patch4_window7_224", 
                    "convnext_large.fb_in22k_ft_in1k"]
    if selected_model_name in large_models:
        tta_chunk = 256
        single_chunk = 512
    else:
        tta_chunk = 512
        single_chunk = 1024
    print(f"Chunk sizes: TTA={tta_chunk}, single={single_chunk}")
    
    # Extract features
    print("Extracting features from training base set...")
    X_train_base = extract_features(model, train_loader_base, device, is_tta=False, chunk_size=single_chunk)
    print("Extracting features from training flip set...")
    X_train_flip = extract_features(model, train_loader_flip, device, is_tta=False, chunk_size=single_chunk)
    print("Extracting features from validation TTA set...")
    X_val = extract_features(model, val_loader_tta, device, is_tta=True, chunk_size=tta_chunk)
    print("Extracting features from test TTA set...")
    X_test = extract_features(model, test_loader_tta, device, is_tta=True, chunk_size=tta_chunk)
    
    # Combine training features (base + flip augmentation)
    X_train = np.vstack([X_train_base, X_train_flip])
    y_train = np.concatenate([tr_labels, tr_labels])
    print(f"Training features shape: {X_train.shape}, validation shape: {X_val.shape}, test shape: {X_test.shape}")
    
    # L2 normalization
    X_train = l2_normalize(X_train)
    X_val = l2_normalize(X_val)
    X_test = l2_normalize(X_test)
    
    # Train Logistic Regression with hyperparameter tuning
    best_logloss = np.inf
    best_C = None
    best_clf = None
    
    for C in args.C_candidates:
        clf = LogisticRegression(penalty='l2', C=C, solver='lbfgs', max_iter=1000, random_state=args.seed)
        clf.fit(X_train, y_train)
        val_probs = clf.predict_proba(X_val)[:, 1]
        val_probs = np.clip(val_probs, 1e-7, 1-1e-7)
        loss = log_loss(va_labels, val_probs)
        print(f"C={C}: validation log loss = {loss:.6f}")
        if loss < best_logloss:
            best_logloss = loss
            best_C = C
            best_clf = clf
    
    print(f"Best C: {best_C} with log loss {best_logloss:.6f}")
    
    # Platt scaling (calibration) on validation set
    val_logits = best_clf.decision_function(X_val)
    calib = LogisticRegression(penalty='l2', C=1e6, solver='lbfgs', max_iter=1000, random_state=args.seed)
    calib.fit(val_logits.reshape(-1, 1), va_labels)
    val_cal_probs = calib.predict_proba(val_logits.reshape(-1, 1))[:, 1]
    val_cal_probs = np.clip(val_cal_probs, 1e-7, 1-1e-7)
    val_cal_logloss = log_loss(va_labels, val_cal_probs)
    print(f"Validation log loss after calibration: {val_cal_logloss:.6f}")
    
    # Test predictions
    test_logits = best_clf.decision_function(X_test)
    test_probs = calib.predict_proba(test_logits.reshape(-1, 1))[:, 1]
    test_probs = np.clip(test_probs, 1e-7, 1-1e-7)
    
    # Save submission
    sub_df = pd.DataFrame({'id': test_ids, 'label': test_probs})
    sub_df.to_csv(os.path.join(args.submission_dir, 'submission.csv'), index=False)
    sub_df.to_csv(os.path.join(args.working_dir, 'submission.csv'), index=False)
    print(f"Saved submission file with {len(sub_df)} predictions.")
    print("Done.")


if __name__ == "__main__":
    main()