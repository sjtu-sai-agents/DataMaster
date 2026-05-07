import os
import random
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import csv
import sys

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
INPUT_DIR = './input'
TRAIN_DIR = os.path.join(INPUT_DIR, 'train')
CLEAN_DIR = os.path.join(INPUT_DIR, 'train_cleaned')
TEST_DIR = os.path.join(INPUT_DIR, 'test')
SUBMISSION_DIR = './submission'
WORKING_DIR = './working'
os.makedirs(SUBMISSION_DIR, exist_ok=True)
os.makedirs(WORKING_DIR, exist_ok=True)

SEED = 42
PATCH_SIZE = 256
BATCH_SIZE = 16
BG_KERNEL = 31
EPOCHS = 8
LR = 1e-3
EMA_DECAY = 0.999
VAL_RATIO = 0.15

# ------------------------------------------------------------
# Set seeds for reproducibility
# ------------------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Keep deterministic false for performance
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

set_seed(SEED)

# ------------------------------------------------------------
# Helper functions for images
# ------------------------------------------------------------
def load_grayscale(path, size=None):
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
    pad = kernel_size // 2
    return F.avg_pool2d(x, kernel_size, stride=1, padding=pad, count_include_pad=False)

def sobel_magnitude_channel(x):
    device = x.device
    gx = F.conv2d(x, _sobel_x.to(device), padding=1)
    gy = F.conv2d(x, _sobel_y.to(device), padding=1)
    mag = torch.sqrt(gx**2 + gy**2 + 1e-6)
    return mag

def build_input_with_bg_grad(img, k=31):
    # img: (B,1,H,W)
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
# Padding utilities for the U‑Net (needs multiples of 16)
# ------------------------------------------------------------
def pad_to_multiple_tensor(x, divisor=16):
    _, _, h, w = x.size()
    pad_h = (divisor - h % divisor) % divisor
    pad_w = (divisor - w % divisor) % divisor
    if pad_h == 0 and pad_w == 0:
        return x, (0,0,0,0)
    # (left, right, top, bottom)
    padding = (pad_w // 2, pad_w - pad_w//2, pad_h // 2, pad_h - pad_h//2)
    x_padded = F.pad(x, padding, mode='reflect')
    return x_padded, padding

def crop_from_padding_tensor(x, padding):
    if padding == (0,0,0,0):
        return x
    left, right, top, bottom = padding
    h, w = x.size()[2], x.size()[3]
    return x[:, :, top:h-bottom, left:w-right]

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
# U‑Net Model (small, residual learning)
# ------------------------------------------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.down(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # padding to match sizes (possible due to odd input dimensions)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNetSmall(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, base=32):
        super().__init__()
        self.inc = DoubleConv(n_channels, base)
        self.down1 = Down(base, base*2)
        self.down2 = Down(base*2, base*4)
        self.down3 = Down(base*4, base*8)
        self.up1 = Up(base*8 + base*4, base*4)
        self.up2 = Up(base*4 + base*2, base*2)
        self.up3 = Up(base*2 + base, base)
        self.outc = nn.Conv2d(base, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        return self.outc(x)

# ------------------------------------------------------------
# Exponential Moving Average (EMA)
# ------------------------------------------------------------
class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name].clone()
        self.backup = {}

# ------------------------------------------------------------
# Prediction functions (with and without TTA)
# ------------------------------------------------------------
def predict_clean(model, img_tensor, use_tta=False):
    """img_tensor: (1,1,H,W) on the same device as model"""
    model.eval()
    with torch.no_grad():
        if use_tta:
            return predict_with_tta_dihedral(model, img_tensor)
        else:
            img_padded, padding = pad_to_multiple_tensor(img_tensor)
            inp = build_input_with_bg_grad(img_padded)   # (1,3,H,W)
            pred_res = model(inp)                       # (1,1,H,W)
            cleaned = img_padded - pred_res
            cleaned = crop_from_padding_tensor(cleaned, padding)
            cleaned = torch.clamp(cleaned, 0, 1)
            return cleaned

def predict_with_tta_dihedral(model, img_tensor):
    img_padded, padding = pad_to_multiple_tensor(img_tensor)
    B, C, H, W = img_padded.shape
    assert B == 1
    accum = torch.zeros_like(img_padded)
    count = 0
    for rot in [0, 90, 180, 270]:
        for flip in [False, True]:
            x = img_padded
            if flip:
                x = torch.flip(x, dims=[-1])
            if rot != 0:
                k = rot // 90
                x = torch.rot90(x, k=k, dims=[-2, -1])
            inp = build_input_with_bg_grad(x)
            pred_res = model(inp)
            cleaned = x - pred_res
            # inverse transformation
            if rot != 0:
                cleaned = torch.rot90(cleaned, k=4-k, dims=[-2, -1])
            if flip:
                cleaned = torch.flip(cleaned, dims=[-1])
            accum += cleaned
            count += 1
    cleaned_avg = accum / count
    cleaned_avg = crop_from_padding_tensor(cleaned_avg, padding)
    cleaned_avg = torch.clamp(cleaned_avg, 0, 1)
    return cleaned_avg

# ------------------------------------------------------------
# Validation RMSE evaluation
# ------------------------------------------------------------
def evaluate_rmse(model, val_data, device):
    model.eval()
    total_sq = 0.0
    total_pixels = 0
    with torch.no_grad():
        for img, tgt, _ in val_data:
            tx = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0).to(device)
            ty = torch.from_numpy(tgt).float().unsqueeze(0).unsqueeze(0).to(device)
            pred = predict_clean(model, tx, use_tta=False)
            diff = pred - ty
            total_sq += (diff ** 2).sum().item()
            total_pixels += diff.numel()
    rmse = np.sqrt(total_sq / total_pixels) if total_pixels > 0 else float('inf')
    return rmse

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ---------- Load training data ----------
    train_files = [f for f in os.listdir(TRAIN_DIR) if f.endswith('.png')]
    pairs = []
    for f in train_files:
        dirty_path = os.path.join(TRAIN_DIR, f)
        clean_path = os.path.join(CLEAN_DIR, f)
        if os.path.exists(clean_path):
            pairs.append((dirty_path, clean_path, f[:-4]))
    print(f"Found {len(pairs)} training pairs.")

    images, clean_images, ids = [], [], []
    for dirty_path, clean_path, img_id in pairs:
        x = load_grayscale(dirty_path)
        y = load_grayscale(clean_path)
        if x.shape != y.shape:
            # resize clean to match dirty
            y = load_grayscale(clean_path, size=(x.shape[1], x.shape[0]))
        images.append(x)
        clean_images.append(y)
        ids.append(img_id)

    # ---------- Train/Validation split ----------
    n_total = len(images)
    indices = list(range(n_total))
    random.shuffle(indices)
    n_val = max(1, int(n_total * VAL_RATIO))
    val_idx = set(indices[:n_val])
    train_data = [(images[i], clean_images[i], ids[i]) for i in indices if i not in val_idx]
    val_data = [(images[i], clean_images[i], ids[i]) for i in indices if i in val_idx]
    print(f"Training samples: {len(train_data)}, Validation samples: {len(val_data)}")

    # ---------- DataLoaders ----------
    def worker_init_fn(worker_id):
        worker_seed = SEED + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)

    num_workers = max(4, min(8, os.cpu_count() or 4))
    train_dataset = PatchDataset(train_data, patch_size=PATCH_SIZE, augment=True, bg_kernel=BG_KERNEL)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True,
                              worker_init_fn=worker_init_fn)

    # ---------- Model, optimizer, loss, EMA ----------
    model = UNetSmall(n_channels=3, n_classes=1, base=32).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    ema = EMA(model, decay=EMA_DECAY)

    # ---------- Training loop ----------
    best_rmse = float('inf')
    best_shadow = None

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x: x

    for epoch in range(EPOCHS):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        total_loss = 0.0
        for inp, target, _ in pbar:
            inp = inp.to(device, non_blocking=True)          # (B,3,H,W)
            target = target.to(device, non_blocking=True)    # (B,1,H,W)
            orig = inp[:, 0:1, :, :]                         # original channel

            optimizer.zero_grad()
            pred_res = model(inp)
            cleaned = orig - pred_res
            loss = criterion(cleaned, target)
            loss.backward()
            optimizer.step()
            ema.update()

            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} average loss: {avg_loss:.6f}")

        # ---------- Validation ----------
        ema.apply_shadow()
        val_rmse = evaluate_rmse(model, val_data, device)
        ema.restore()
        print(f"Validation RMSE: {val_rmse:.6f}")

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_shadow = {k: v.clone() for k, v in ema.shadow.items()}
            # Save checkpoint
            torch.save({
                'model_state_dict': model.state_dict(),
                'ema_shadow': ema.shadow,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_rmse': val_rmse,
                'epoch': epoch,
            }, os.path.join(WORKING_DIR, 'best_model.pth'))
            print("Best model saved.")

    # ---------- Load best EMA weights ----------
    if best_shadow is not None:
        for name, param in model.named_parameters():
            if name in best_shadow:
                param.data = best_shadow[name].to(param.device)
    else:
        # Fallback: use the last EMA state
        ema.apply_shadow()
    model.eval()

    # ---------- Final validation RMSE (the metric to print) ----------
    final_rmse = evaluate_rmse(model, val_data, device)
    print(f"\nFinal validation RMSE: {final_rmse:.6f}")

    # ---------- Generate submission ----------
    test_files = [f for f in os.listdir(TEST_DIR) if f.endswith('.png')]
    test_files.sort(key=lambda x: int(x.split('.')[0]))
    sub_path = os.path.join(SUBMISSION_DIR, 'submission.csv')

    with open(sub_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'value'])
        for fname in test_files:
            tid = fname.split('.')[0]
            img_arr = load_grayscale(os.path.join(TEST_DIR, fname))
            tx = torch.from_numpy(img_arr).float().unsqueeze(0).unsqueeze(0).to(device)
            pred = predict_clean(model, tx, use_tta=True)
            pred_np = pred.squeeze().cpu().numpy()   # (H,W)
            H, W = pred_np.shape
            for r in range(H):
                for c in range(W):
                    writer.writerow([f"{tid}_{r+1}_{c+1}", f"{pred_np[r,c]:.6f}"])
            print(f"Processed test image {tid}")

    print(f"Submission saved to {sub_path}")

if __name__ == "__main__":
    main()