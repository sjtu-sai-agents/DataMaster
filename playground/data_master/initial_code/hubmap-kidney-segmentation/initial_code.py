import os
import random
import gc
from glob import glob
from tqdm import tqdm

import numpy as np
import pandas as pd
import cv2
import tifffile
from skimage import morphology

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import albumentations as A
import segmentation_models_pytorch as smp

# ----------------------------------------------------------------------
# Reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed()

# ----------------------------------------------------------------------
# RLE encoding / decoding
def rle_decode(mask_rle, shape):
    if pd.isna(mask_rle) or mask_rle == '':
        return np.zeros(shape, dtype=np.uint8)
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order='F')

def rle_encode(img):
    pixels = img.flatten(order='F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

# ----------------------------------------------------------------------
# Image loading (handles various TIFF formats)
def load_image_file(img_path):
    img = tifffile.imread(img_path)
    img = np.squeeze(img)                     # remove singleton dimensions
    if img.ndim == 2:                         # grayscale -> stack to 3 channels
        img = np.stack([img, img, img], axis=-1)
    elif img.ndim == 3:
        if img.shape[0] == 3:                 # channels first -> HWC
            img = img.transpose(1, 2, 0)
        elif img.shape[2] != 3:               # unknown 3D -> convert to grayscale and stack
            img = np.mean(img, axis=2, keepdims=False)
            img = np.stack([img, img, img], axis=-1)
    # ensure uint8
    if img.dtype == np.uint16:
        img = (img // 256).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8) if img.max() <= 1 else img.astype(np.uint8)
    return img

# ----------------------------------------------------------------------
# Load training data from CSV and TIFFs
def load_train_data(train_csv, train_dir):
    df = pd.read_csv(train_csv)
    data = {}
    for _, row in df.iterrows():
        img_id = row['id']
        img_path = os.path.join(train_dir, f"{img_id}.tiff")
        print(f"Loading {img_id} ...")
        img = load_image_file(img_path)
        mask = rle_decode(row['encoding'], (img.shape[0], img.shape[1]))
        coords = np.argwhere(mask > 0)
        data[img_id] = {
            'image': img,
            'mask': mask,
            'coords': coords,
            'shape': img.shape
        }
    return data

# ----------------------------------------------------------------------
# Constants
PATCH_SIZE = 1024
RESIZE_SIZE = 512
SAMPLES_PER_EPOCH = 2000
BATCH_SIZE = 8
EPOCHS = 25
LR = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = 4

# ----------------------------------------------------------------------
# Dataset – online patch sampling
class KidneyDataset(Dataset):
    def __init__(self, data_dict, patch_size=PATCH_SIZE, resize_size=RESIZE_SIZE,
                 transform=None, training=True):
        self.data_dict = data_dict
        self.keys = list(data_dict.keys())
        self.patch_size = patch_size
        self.resize_size = resize_size
        self.transform = transform
        self.training = training

    def __len__(self):
        return SAMPLES_PER_EPOCH

    def __getitem__(self, idx):
        # random image
        key = random.choice(self.keys)
        info = self.data_dict[key]
        img = info['image']
        mask = info['mask']
        h, w = img.shape[:2]

        # centre selection: 80% near a glomerulus, else random
        if self.training and random.random() < 0.8 and len(info['coords']) > 0:
            coord = random.choice(info['coords'])
            yc, xc = coord[0], coord[1]
            yc += random.randint(-100, 100)
            xc += random.randint(-100, 100)
        else:
            yc = random.randint(self.patch_size//2, h - self.patch_size//2 - 1)
            xc = random.randint(self.patch_size//2, w - self.patch_size//2 - 1)

        y0 = max(0, min(yc - self.patch_size//2, h - self.patch_size))
        x0 = max(0, min(xc - self.patch_size//2, w - self.patch_size))
        y1 = y0 + self.patch_size
        x1 = x0 + self.patch_size

        patch_img = img[y0:y1, x0:x1, :]
        patch_mask = mask[y0:y1, x0:x1]

        # augmentations
        if self.transform:
            transformed = self.transform(image=patch_img, mask=patch_mask)
            patch_img = transformed['image']
            patch_mask = transformed['mask']

        # resize
        patch_img = cv2.resize(patch_img, (self.resize_size, self.resize_size), interpolation=cv2.INTER_LINEAR)
        patch_mask = cv2.resize(patch_mask, (self.resize_size, self.resize_size), interpolation=cv2.INTER_NEAREST)

        # normalize image
        patch_img = patch_img.astype(np.float32) / 255.0

        # to tensor
        patch_img = torch.from_numpy(patch_img).permute(2, 0, 1).float()
        patch_mask = torch.from_numpy(patch_mask).unsqueeze(0).float()
        return patch_img, patch_mask

# ----------------------------------------------------------------------
# Data augmentation
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
    A.ElasticTransform(p=0.2),
    A.GridDistortion(p=0.2),
    A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=0.2),
    A.CLAHE(p=0.2),
    A.RandomBrightnessContrast(p=0.3),
    A.GaussNoise(p=0.2),
    A.Blur(p=0.2),
])

# ----------------------------------------------------------------------
# Load data and split (last image for validation)
INPUT_DIR = "./input"
TRAIN_CSV = os.path.join(INPUT_DIR, "train.csv")
TRAIN_IMG_DIR = os.path.join(INPUT_DIR, "train")
TEST_IMG_DIR = os.path.join(INPUT_DIR, "test")
SUBMISSION_DIR = "./submission"
os.makedirs(SUBMISSION_DIR, exist_ok=True)
WORKING_DIR = "./working"
os.makedirs(WORKING_DIR, exist_ok=True)

print("Loading training data...")
all_data = load_train_data(TRAIN_CSV, TRAIN_IMG_DIR)
keys = list(all_data.keys())
val_id = keys[-1]
train_ids = keys[:-1]
train_data = {k: all_data[k] for k in train_ids}
val_data = {val_id: all_data[val_id]}

# ----------------------------------------------------------------------
# Model, loss, optimizer
model = smp.UnetPlusPlus(
    encoder_name='efficientnet-b5',
    encoder_weights='imagenet',
    in_channels=3,
    classes=1,
    activation=None
).to(DEVICE)

dice_loss = smp.losses.DiceLoss(mode='binary', from_logits=True)
lovasz_loss = smp.losses.LovaszLoss(mode='binary', from_logits=True)

def criterion(y_pred, y_true):
    return 0.5 * dice_loss(y_pred, y_true) + 0.5 * lovasz_loss(y_pred, y_true)

optimizer = optim.AdamW(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

# ----------------------------------------------------------------------
# Training
train_dataset = KidneyDataset(train_data, transform=train_transform, training=True)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)

use_amp = DEVICE.type == 'cuda'
if use_amp:
    scaler = torch.cuda.amp.GradScaler()

print("Starting training...")
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    progress = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}')
    for images, masks in progress:
        images = images.to(DEVICE, non_blocking=True)
        masks = masks.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        if use_amp:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

        epoch_loss += loss.item()
        progress.set_postfix(loss=loss.item())

    avg_loss = epoch_loss / len(progress)
    print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")
    scheduler.step()

# ----------------------------------------------------------------------
# Inference utilities (TTA, sliding window)
tta_transforms = [
    (lambda x: x, lambda x: x),
    (lambda x: torch.flip(x, dims=[3]), lambda x: torch.flip(x, dims=[3])),
    (lambda x: torch.flip(x, dims=[2]), lambda x: torch.flip(x, dims=[2])),
    (lambda x: torch.rot90(x, k=1, dims=[2,3]), lambda x: torch.rot90(x, k=3, dims=[2,3])),
    (lambda x: torch.rot90(x, k=2, dims=[2,3]), lambda x: torch.rot90(x, k=2, dims=[2,3])),
    (lambda x: torch.rot90(x, k=3, dims=[2,3]), lambda x: torch.rot90(x, k=1, dims=[2,3])),
]

def predict_patch_with_tta(model, patch, device):
    """patch: (H,W,3) float32 in [0,1] -> returns prob (H,W)"""
    model.eval()
    with torch.no_grad():
        inp = torch.from_numpy(patch).permute(2,0,1).float().unsqueeze(0).to(device)
        logits = 0
        for tta_fn, inv_fn in tta_transforms:
            transformed = tta_fn(inp)
            pred = model(transformed)
            pred = inv_fn(pred)
            logits += pred
        logits /= len(tta_transforms)
        prob = torch.sigmoid(logits).squeeze().cpu().numpy()
        return prob

def predict_full_image(model, image, device):
    """image: uint8 (H,W,3) -> prob map (H,W)"""
    h, w = image.shape[:2]
    prob_map = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)

    step = PATCH_SIZE // 2
    y_starts = list(range(0, h - PATCH_SIZE + 1, step))
    if y_starts[-1] + PATCH_SIZE < h:
        y_starts.append(h - PATCH_SIZE)
    x_starts = list(range(0, w - PATCH_SIZE + 1, step))
    if x_starts[-1] + PATCH_SIZE < w:
        x_starts.append(w - PATCH_SIZE)

    img_float = image.astype(np.float32) / 255.0
    model.eval()
    with torch.no_grad():
        for y in tqdm(y_starts, desc="Sliding window"):
            for x in x_starts:
                patch = img_float[y:y+PATCH_SIZE, x:x+PATCH_SIZE, :]
                patch_resized = cv2.resize(patch, (RESIZE_SIZE, RESIZE_SIZE), interpolation=cv2.INTER_LINEAR)
                prob_patch = predict_patch_with_tta(model, patch_resized, device)
                prob_patch_full = cv2.resize(prob_patch, (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_LINEAR)
                prob_map[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += prob_patch_full
                count_map[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += 1

    prob_map = np.divide(prob_map, count_map, out=np.zeros_like(prob_map), where=count_map>0)
    return prob_map

def post_process(prob, threshold=0.5, min_size=500):
    mask = (prob > threshold).astype(np.uint8)
    mask = morphology.remove_small_objects(mask.astype(bool), min_size=min_size, connectivity=2)
    return mask.astype(np.uint8)

def compute_dice(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    if pred.sum() == 0 and gt.sum() == 0:
        return 1.0
    return (2.0 * inter) / (pred.sum() + gt.sum())

# ----------------------------------------------------------------------
# Validation (print metric)
print("\nEvaluating on validation image...")
val_id = list(val_data.keys())[0]
val_img = val_data[val_id]['image']
val_mask = val_data[val_id]['mask']
val_prob = predict_full_image(model, val_img, DEVICE)
val_pred = post_process(val_prob)
val_dice = compute_dice(val_pred, val_mask)
print(f"Validation Dice: {val_dice:.4f}")

# ----------------------------------------------------------------------
# Test prediction and submission
test_files = glob(os.path.join(TEST_IMG_DIR, "*.tiff"))
test_ids = [os.path.basename(f).replace('.tiff', '') for f in test_files]
submission = []

for test_id in test_ids:
    print(f"\nProcessing test image {test_id} ...")
    img_path = os.path.join(TEST_IMG_DIR, f"{test_id}.tiff")
    test_img = load_image_file(img_path)
    test_prob = predict_full_image(model, test_img, DEVICE)
    test_pred = post_process(test_prob)
    rle = rle_encode(test_pred) if np.any(test_pred) else ''
    submission.append((test_id, rle))

    del test_img, test_prob, test_pred
    gc.collect()
    torch.cuda.empty_cache()

sub_df = pd.DataFrame(submission, columns=['id', 'predicted'])
sub_df.to_csv(os.path.join(SUBMISSION_DIR, 'submission.csv'), index=False)
print("\nSubmission saved to ./submission/submission.csv")