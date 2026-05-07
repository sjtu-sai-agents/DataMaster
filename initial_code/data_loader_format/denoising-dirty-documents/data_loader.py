import os
import random
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import pandas as pd

# ------------------------------------------------------------
# Helper functions for images
# ------------------------------------------------------------
def load_grayscale(path, size=None):
    """Load image as grayscale numpy array normalized to [0, 1]."""
    img = Image.open(path).convert('L')
    if size is not None:
        img = img.resize(size, Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr

# ------------------------------------------------------------
# Feature engineering (local mean & Sobel magnitude)
# ------------------------------------------------------------
_sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32).view(1,1,3,3)
_sobel_y = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32).view(1,1,3,3)

def local_mean_channel(x, kernel_size=31):
    """Compute local mean using average pooling."""
    pad = kernel_size // 2
    return F.avg_pool2d(x, kernel_size, stride=1, padding=pad, count_include_pad=False)

def sobel_magnitude_channel(x):
    """Compute Sobel gradient magnitude."""
    device = x.device
    gx = F.conv2d(x, _sobel_x.to(device), padding=1)
    gy = F.conv2d(x, _sobel_y.to(device), padding=1)
    mag = torch.sqrt(gx**2 + gy**2 + 1e-6)
    return mag

def build_input_with_bg_grad(img, k=31):
    """Build 3-channel input: original, local mean, gradient magnitude."""
    lmean = local_mean_channel(img, k)
    gmag = sobel_magnitude_channel(img)
    B = gmag.shape[0]
    for i in range(B):
        gmin = gmag[i].min()
        gmax = gmag[i].max()
        if gmax - gmin > 0:
            gmag[i] = (gmag[i] - gmin) / (gmax - gmin)
        else:
            gmag[i] = torch.zeros_like(gmag[i])
    return torch.cat([img, lmean, gmag], dim=1)   # (B,3,H,W)

# ------------------------------------------------------------
# Dataset for training (random patches + augmentation)
# ------------------------------------------------------------
class PatchDataset(Dataset):
    def __init__(self, data, patch_size=256, augment=True, bg_kernel=31):
        self.data = data                # list of (dirty, clean, id)
        self.patch_size = patch_size
        self.augment = augment
        self.bg_kernel = bg_kernel
        self.length = max(1, len(data) * 200)   # oversample

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # randomly choose one image from the data
        img, tgt, _ = random.choice(self.data)
        H, W = img.shape
        ps = min(self.patch_size, H, W)
        top = random.randint(0, max(0, H - ps))
        left = random.randint(0, max(0, W - ps))
        patch_x = img[top:top+ps, left:left+ps]
        patch_y = tgt[top:top+ps, left:left+ps]

        # Pad to patch_size if needed
        if ps < self.patch_size:
            pad_h = self.patch_size - ps
            pad_w = self.patch_size - ps
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            patch_x = np.pad(patch_x, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='reflect')
            patch_y = np.pad(patch_y, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='reflect')

        # Data augmentation (dihedral)
        if self.augment:
            if random.random() < 0.5:   # horizontal flip
                patch_x = np.flip(patch_x, axis=1).copy()
                patch_y = np.flip(patch_y, axis=1).copy()
            if random.random() < 0.5:   # vertical flip
                patch_x = np.flip(patch_x, axis=0).copy()
                patch_y = np.flip(patch_y, axis=0).copy()
            k = random.randint(0, 3)    # rotation by 90*k degrees
            if k != 0:
                patch_x = np.rot90(patch_x, k=k).copy()
                patch_y = np.rot90(patch_y, k=k).copy()

        # Convert to torch tensors
        tx = torch.from_numpy(patch_x).float().unsqueeze(0).unsqueeze(0)   # (1,1,H,W)
        ty = torch.from_numpy(patch_y).float().unsqueeze(0).unsqueeze(0)   # (1,1,H,W)

        # Build 3‑channel input
        inp = build_input_with_bg_grad(tx, k=self.bg_kernel)[0]   # (3,H,W)
        target = ty[0]                                            # (1,H,W)
        tres = tx[0] - ty[0]                                      # (1,H,W) – not used later
        return inp, target, tres

# ------------------------------------------------------------
# MyDataLoader class
# ------------------------------------------------------------
class MyDataLoader(BaseDataLoader):
    def __init__(self, input_dir='./input', val_ratio=0.15, seed=42, **kwargs):
        super().__init__(**kwargs)
        self.input_dir = input_dir
        self.train_dir = os.path.join(input_dir, 'train')
        self.clean_dir = os.path.join(input_dir, 'train_cleaned')
        self.test_dir = os.path.join(input_dir, 'test')
        self.val_ratio = val_ratio
        self.seed = seed

    def setup(self):
        """
        Load data, feature engineering, data augmentation, etc.
        Sets self.train_data and self.test_data (validation data).
        """
        # Load training data
        train_files = [f for f in os.listdir(self.train_dir) if f.endswith('.png')]
        pairs = []
        for f in train_files:
            dirty_path = os.path.join(self.train_dir, f)
            clean_path = os.path.join(self.clean_dir, f)
            if os.path.exists(clean_path):
                pairs.append((dirty_path, clean_path, f[:-4]))
        
        images, clean_images, ids = [], [], []
        for dirty_path, clean_path, img_id in pairs:
            x = load_grayscale(dirty_path)
            y = load_grayscale(clean_path)
            if x.shape != y.shape:
                y = load_grayscale(clean_path, size=(x.shape[1], x.shape[0]))
            images.append(x)
            clean_images.append(y)
            ids.append(img_id)
        
        # Train/Validation split - check for val.csv first
        val_csv_path = os.path.join(self.input_dir, 'val.csv')
        if os.path.exists(val_csv_path):
            val_df = pd.read_csv(val_csv_path)
            # Check for 'image' column or use first column
            if 'image' in val_df.columns:
                val_ids = set(val_df['image'].values)
            else:
                val_ids = set(val_df.iloc[:, 0].values)
            train_data = [(images[i], clean_images[i], ids[i]) for i in range(len(ids)) if ids[i] not in val_ids]
            val_data = [(images[i], clean_images[i], ids[i]) for i in range(len(ids)) if ids[i] in val_ids]
        else:
            # Fallback to random split if val.csv doesn't exist
            random.seed(self.seed)
            n_total = len(images)
            indices = list(range(n_total))
            random.shuffle(indices)
            n_val = max(1, int(n_total * self.val_ratio))
            val_idx = set(indices[:n_val])
            train_data = [(images[i], clean_images[i], ids[i]) for i in indices if i not in val_idx]
            val_data = [(images[i], clean_images[i], ids[i]) for i in indices if i in val_idx]
        
        self.train_data = train_data
        self.test_data = val_data  # Using val_data as test_data for validation

    def describe(self) -> str:
        """
        Return a description of your data processing approach.
        """
        return ("Data loader for image denoising/cleaning task. "
                "Loads grayscale images from train/train_cleaned directories. "
                "Supports pre-split validation set from val.csv if available, "
                "otherwise uses random split with configurable ratio. "
                "Provides PatchDataset for patch-based training with data augmentation "
                "(horizontal/vertical flips, 90-degree rotations). "
                "Uses local mean and Sobel gradient magnitude as additional input channels for feature engineering.")