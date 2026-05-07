import os
import json
import random
import warnings
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
from sklearn.model_selection import StratifiedShuffleSplit
import timm
from timm.utils import ModelEmaV2
from timm.loss import SoftTargetCrossEntropy
from timm.data import Mixup
from timm.optim import AdamW
from timm.scheduler import CosineLRScheduler

warnings.filterwarnings('ignore')

# ---------- Configuration ----------
DATA_DIR = "input"
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
LABEL_MAP = os.path.join(DATA_DIR, "label_num_to_disease_map.json")
SUBMISSION_PATH = "submission/submission.csv"
WORKING_DIR = "working"
CHECKPOINT_PATH = os.path.join(WORKING_DIR, "best_convnext_base_two_stage.pth")

VAL_RATIO = 0.15
BATCH_SIZE = 12
NUM_WORKERS = 6
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

STAGE1 = {'size': 384, 'epochs': 3, 'base_lr': 2e-4, 'min_lr': 1e-5}
STAGE2 = {'size': 448, 'epochs': 2, 'base_lr': 1e-4, 'min_lr': 5e-6}

MIXUP_ARGS = {
    'mixup_alpha': 0.4,
    'cutmix_alpha': 1.0,
    'prob': 0.6,
    'switch_prob': 0.5,
    'label_smoothing': 0.1,
    'num_classes': 5
}

VAL_SCALES = [384, 448]   # multi-scale evaluation
FLIP_TTA = True

os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

# ---------- Reproducibility ----------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

set_seed(42)

# ---------- Dataset ----------
class CassavaDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, is_test=False):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['image_id']
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        if self.is_test:
            return image, img_name
        label = self.df.iloc[idx]['label']
        return image, torch.tensor(label, dtype=torch.long)

# ---------- Transforms ----------
def get_train_transform(size):
    return T.Compose([
        T.RandomResizedCrop(size, scale=(0.6, 1.0), interpolation=InterpolationMode.BILINEAR),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandAugment(num_ops=2, magnitude=10, interpolation=InterpolationMode.BILINEAR),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value='random')
    ])

def get_eval_transform(size):
    resize_size = int(size * 1.15)
    return T.Compose([
        T.Resize(resize_size, interpolation=InterpolationMode.BILINEAR),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

# ---------- Data Loaders ----------
def build_train_loader(df, size, batch_size=BATCH_SIZE, shuffle=True):
    transform = get_train_transform(size)
    dataset = CassavaDataset(df, TRAIN_IMG_DIR, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    return loader

def build_eval_loader(df, size, batch_size=BATCH_SIZE, shuffle=False):
    transform = get_eval_transform(size)
    dataset = CassavaDataset(df, TRAIN_IMG_DIR, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                        num_workers=NUM_WORKERS, pin_memory=True)
    return loader

# ---------- Model ----------
def create_model():
    model = timm.create_model('convnext_base.fb_in22k_ft_in1k', pretrained=True, num_classes=5)
    model = model.to(DEVICE)
    return model

# ---------- Multi-scale Evaluation ----------
def evaluate_multi_scale(model, val_df, scales=VAL_SCALES, flip_tta=FLIP_TTA, device=DEVICE):
    model.eval()
    num_samples = len(val_df)
    total_probs = torch.zeros((num_samples, 5), device=device)

    with torch.no_grad():
        for size in scales:
            loader = build_eval_loader(val_df, size)
            start_idx = 0
            for images, _ in loader:
                images = images.to(device)
                batch_size = images.size(0)
                with torch.cuda.amp.autocast(enabled=True):
                    outputs = model(images)
                    probs = torch.softmax(outputs, dim=1)
                if flip_tta:
                    images_flip = torch.flip(images, dims=[3])
                    with torch.cuda.amp.autocast(enabled=True):
                        outputs_flip = model(images_flip)
                    probs_flip = torch.softmax(outputs_flip, dim=1)
                    probs = (probs + probs_flip) / 2
                total_probs[start_idx:start_idx+batch_size] += probs
                start_idx += batch_size

    avg_probs = total_probs / len(scales)
    preds = torch.argmax(avg_probs, dim=1).cpu().numpy()
    true_labels = val_df['label'].values
    acc = (preds == true_labels).mean()
    return acc

# ---------- Training Stage ----------
def train_stage(train_df, val_df, model, ema, size, epochs, base_lr, min_lr, checkpoint_path, best_acc=0.0):
    train_loader = build_train_loader(train_df, size)
    optimizer = AdamW(model.parameters(), lr=base_lr, weight_decay=1e-2)
    total_steps = len(train_loader) * epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = CosineLRScheduler(optimizer, t_initial=total_steps, lr_min=min_lr,
                                  warmup_t=warmup_steps, warmup_lr_init=1e-6, warmup_prefix=True)
    mixup_fn = Mixup(**MIXUP_ARGS)
    criterion = SoftTargetCrossEntropy()
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(epochs):
        model.train()
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            images, labels = mixup_fn(images, labels)
            with torch.cuda.amp.autocast(enabled=True):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            ema.update(model)
            step = epoch * len(train_loader) + batch_idx + 1
            scheduler.step(step)

        # Validation after epoch
        acc = evaluate_multi_scale(ema.module, val_df, scales=VAL_SCALES, flip_tta=FLIP_TTA)
        print(f"Epoch {epoch+1}/{epochs} - Validation Accuracy: {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            torch.save({
                'model': model.state_dict(),
                'ema_model': ema.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'best_acc': best_acc,
                'stage_size': size,
            }, checkpoint_path)
            print(f"Saved new best checkpoint with accuracy {best_acc:.4f}")

    return best_acc

# ---------- Main ----------
def main():
    # Load and split data
    train_df = pd.read_csv(TRAIN_CSV)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=VAL_RATIO, random_state=42)
    for train_index, val_index in sss.split(train_df['image_id'], train_df['label']):
        train_data = train_df.iloc[train_index].reset_index(drop=True)
        val_data = train_df.iloc[val_index].reset_index(drop=True)

    print(f"Train size: {len(train_data)}, Validation size: {len(val_data)}")

    # Model and EMA
    model = create_model()
    ema = ModelEmaV2(model, decay=0.999, device=DEVICE)

    best_acc = 0.0

    # Stage 1
    print("\n=== Stage 1: training at size 384 ===\n")
    best_acc = train_stage(train_data, val_data, model, ema,
                           size=STAGE1['size'],
                           epochs=STAGE1['epochs'],
                           base_lr=STAGE1['base_lr'],
                           min_lr=STAGE1['min_lr'],
                           checkpoint_path=CHECKPOINT_PATH,
                           best_acc=best_acc)

    # Load best from Stage 1
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    ema.module.load_state_dict(checkpoint['ema_model'])
    best_acc = checkpoint['best_acc']
    print(f"\nLoaded best checkpoint from Stage 1 with accuracy {best_acc:.4f}")

    # Stage 2
    print("\n=== Stage 2: training at size 448 ===\n")
    best_acc = train_stage(train_data, val_data, model, ema,
                           size=STAGE2['size'],
                           epochs=STAGE2['epochs'],
                           base_lr=STAGE2['base_lr'],
                           min_lr=STAGE2['min_lr'],
                           checkpoint_path=CHECKPOINT_PATH,
                           best_acc=best_acc)

    # Load final best model
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    ema.module.load_state_dict(checkpoint['ema_model'])
    best_acc = checkpoint['best_acc']
    print(f"\nFinal best validation accuracy: {best_acc:.4f}")

    # Recompute validation accuracy (metric to print)
    final_acc = evaluate_multi_scale(ema.module, val_data, scales=VAL_SCALES, flip_tta=FLIP_TTA)
    print(f"Final validation accuracy (recomputed): {final_acc:.4f}")

    # ---------- Test Prediction ----------
    print("\nGenerating test predictions...")
    test_df = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))
    num_test = len(test_df)
    total_probs = torch.zeros((num_test, 5), device=DEVICE)

    ema.module.eval()
    with torch.no_grad():
        for size in VAL_SCALES:
            transform = get_eval_transform(size)
            dataset = CassavaDataset(test_df, TEST_IMG_DIR, transform=transform, is_test=True)
            loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=NUM_WORKERS, pin_memory=True)
            start_idx = 0
            for images, _ in loader:
                images = images.to(DEVICE)
                batch_size = images.size(0)
                with torch.cuda.amp.autocast(enabled=True):
                    outputs = ema.module(images)
                    probs = torch.softmax(outputs, dim=1)
                if FLIP_TTA:
                    images_flip = torch.flip(images, dims=[3])
                    with torch.cuda.amp.autocast(enabled=True):
                        outputs_flip = ema.module(images_flip)
                    probs_flip = torch.softmax(outputs_flip, dim=1)
                    probs = (probs + probs_flip) / 2
                total_probs[start_idx:start_idx+batch_size] += probs
                start_idx += batch_size

    avg_probs = total_probs / len(VAL_SCALES)
    pred_labels = torch.argmax(avg_probs, dim=1).cpu().numpy()
    test_df['label'] = pred_labels
    test_df.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")

if __name__ == '__main__':
    main()