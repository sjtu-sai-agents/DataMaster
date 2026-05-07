import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import csv
import argparse

# ------------------------------------------------------------
# Set seeds for reproducibility
# ------------------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

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
# Argument parser
# ------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description='Image Denoising Training Script')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--patch_size', type=int, default=256, help='Patch size for training')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--bg_kernel', type=int, default=31, help='Background kernel size for local mean computation')
    parser.add_argument('--epochs', type=int, default=8, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--ema_decay', type=float, default=0.999, help='EMA decay rate')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='Validation ratio (used if val.csv not found)')
    parser.add_argument('--input_dir', type=str, default='./input', help='Input directory containing train/test data')
    parser.add_argument('--submission_dir', type=str, default='./submission', help='Directory to save submission files')
    parser.add_argument('--working_dir', type=str, default='./working', help='Working directory for checkpoints')
    return parser.parse_args()

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    args = parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Create directories
    os.makedirs(args.submission_dir, exist_ok=True)
    os.makedirs(args.working_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create data loader and get data
    data_loader = MyDataLoader(
        input_dir=args.input_dir,
        val_ratio=args.val_ratio,
        seed=args.seed
    )
    train_data, val_data = data_loader.get_data()
    print(f"Training samples: {len(train_data)}, Validation samples: {len(val_data)}")

    # DataLoader worker initialization
    def worker_init_fn(worker_id):
        worker_seed = args.seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)

    num_workers = max(4, min(8, os.cpu_count() or 4))
    train_dataset = PatchDataset(train_data, patch_size=args.patch_size, augment=True, bg_kernel=args.bg_kernel)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True,
                              worker_init_fn=worker_init_fn)

    # Model, optimizer, loss, EMA
    model = UNetSmall(n_channels=3, n_classes=1, base=32).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    ema = EMA(model, decay=args.ema_decay)

    # Training loop
    best_rmse = float('inf')
    best_shadow = None

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x: x

    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
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

        # Validation
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
            }, os.path.join(args.working_dir, 'best_model.pth'))
            print("Best model saved.")

    # Load best EMA weights
    if best_shadow is not None:
        for name, param in model.named_parameters():
            if name in best_shadow:
                param.data = best_shadow[name].to(param.device)
    else:
        # Fallback: use the last EMA state
        ema.apply_shadow()
    model.eval()

    # Final validation RMSE
    final_rmse = evaluate_rmse(model, val_data, device)
    print(f"\nFinal validation RMSE: {final_rmse:.6f}")

    # Generate submission
    test_dir = os.path.join(args.input_dir, 'test')
    test_files = [f for f in os.listdir(test_dir) if f.endswith('.png')]
    test_files.sort(key=lambda x: int(x.split('.')[0]))
    sub_path = os.path.join(args.submission_dir, 'submission.csv')

    with open(sub_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'value'])
        for fname in test_files:
            tid = fname.split('.')[0]
            img_arr = load_grayscale(os.path.join(test_dir, fname))
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