import os
import sys
import random
import json
import math
import numpy as np
import pandas as pd
import cv2
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import joblib
from tqdm import tqdm
from collections import defaultdict

import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader
from torch import optim
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

# Constants
SEED = 42
IMG_SIZE = 800
BATCH_SIZE = 8
NUM_WORKERS = 4
NUM_EPOCHS = 8
VAL_SPLIT = 0.2
CONF_THRESH = 0.1  # for test predictions
IOU_THRESH = 0.4   # for mAP

# Paths
INPUT_DIR = "./input"
TRAIN_CSV = os.path.join(INPUT_DIR, "train.csv")
TRAIN_DICOM_DIR = os.path.join(INPUT_DIR, "train")
TEST_DICOM_DIR = os.path.join(INPUT_DIR, "test")

WORKING_DIR = "./working"
os.makedirs(WORKING_DIR, exist_ok=True)
TRAIN_JPG_DIR = os.path.join(WORKING_DIR, "train_jpg")
TEST_JPG_DIR = os.path.join(WORKING_DIR, "test_jpg")
os.makedirs(TRAIN_JPG_DIR, exist_ok=True)
os.makedirs(TEST_JPG_DIR, exist_ok=True)

SUBMISSION_DIR = "./submission"
os.makedirs(SUBMISSION_DIR, exist_ok=True)
SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

META_FILE = os.path.join(WORKING_DIR, "meta.json")

# Set seeds for reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_and_preprocess_dicom(dcm_path, output_dir, img_size=IMG_SIZE):
    """Read a DICOM file, apply VOI LUT, convert to uint8, resize, save as JPG, return original dimensions."""
    image_id = os.path.splitext(os.path.basename(dcm_path))[0]
    try:
        ds = pydicom.dcmread(dcm_path, force=True)
        # Apply VOI LUT (if available) to get correct contrast
        data = apply_voi_lut(ds.pixel_array, ds)
        # Handle MONOCHROME1 (inverted) images
        if ds.PhotometricInterpretation == "MONOCHROME1":
            data = np.amax(data) - data
        # Normalize to 0-255
        data = data - np.min(data)
        if np.max(data) > 0:
            data = data / np.max(data)
            data = (data * 255).astype(np.uint8)
        else:
            data = np.zeros(data.shape, dtype=np.uint8)
        # Resize
        data = cv2.resize(data, (img_size, img_size), interpolation=cv2.INTER_AREA)
        # Convert to 3-channel RGB
        if data.ndim == 2:
            data = cv2.cvtColor(data, cv2.COLOR_GRAY2RGB)
        else:
            # If already multi-channel, assume it is BGR (OpenCV default) and convert to RGB
            data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)

        # Save as JPG (OpenCV expects BGR)
        out_path = os.path.join(output_dir, f"{image_id}.jpg")
        data_bgr = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
        cv2.imwrite(out_path, data_bgr)

        # Original dimensions from DICOM
        orig_h = int(ds.Rows)
        orig_w = int(ds.Columns)
        return image_id, (orig_h, orig_w)

    except Exception as e:
        print(f"Error processing {dcm_path}: {e}")
        # Create dummy image and return dummy dimensions
        dummy = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        out_path = os.path.join(output_dir, f"{image_id}.jpg")
        cv2.imwrite(out_path, dummy)
        orig_h, orig_w = img_size, img_size
        return image_id, (orig_h, orig_w)

def preprocess_all():
    """Convert all DICOM files to JPG and collect original dimensions."""
    meta = {}
    # Process train
    train_files = [f for f in os.listdir(TRAIN_DICOM_DIR) if f.endswith('.dicom')]
    print(f"Processing {len(train_files)} train DICOMs...")
    results = joblib.Parallel(n_jobs=4)(
        joblib.delayed(load_and_preprocess_dicom)(
            os.path.join(TRAIN_DICOM_DIR, f), TRAIN_JPG_DIR, IMG_SIZE
        ) for f in tqdm(train_files)
    )
    for img_id, dims in results:
        meta[img_id] = dims
    # Process test
    test_files = [f for f in os.listdir(TEST_DICOM_DIR) if f.endswith('.dicom')]
    print(f"Processing {len(test_files)} test DICOMs...")
    results = joblib.Parallel(n_jobs=4)(
        joblib.delayed(load_and_preprocess_dicom)(
            os.path.join(TEST_DICOM_DIR, f), TEST_JPG_DIR, IMG_SIZE
        ) for f in tqdm(test_files)
    )
    for img_id, dims in results:
        meta[img_id] = dims
    # Save meta
    with open(META_FILE, 'w') as f:
        json.dump(meta, f)
    return meta

def load_meta():
    """Load meta data from JSON, or preprocess if not exists."""
    if os.path.exists(META_FILE):
        with open(META_FILE, 'r') as f:
            meta = json.load(f)
        return meta
    else:
        return preprocess_all()

# IoU computation
def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / (union + 1e-6)

# Dataset class
class VinBigDataDataset(Dataset):
    def __init__(self, dataframe, image_dir, meta_dict, transforms=False):
        self.df = dataframe
        self.image_dir = image_dir
        self.meta = meta_dict
        self.transforms = transforms
        self.image_ids = dataframe['image_id'].unique()

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        # Load image
        img_path = os.path.join(self.image_dir, f"{image_id}.jpg")
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0  # to [0,1]
        else:
            img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)

        # Original dimensions
        orig_h, orig_w = self.meta.get(image_id, (IMG_SIZE, IMG_SIZE))
        scale_x = IMG_SIZE / orig_w
        scale_y = IMG_SIZE / orig_h

        # Get annotations for this image
        records = self.df[self.df['image_id'] == image_id]
        boxes = []
        labels = []
        for _, row in records.iterrows():
            class_id = row['class_id']
            if class_id == 14:  # skip "No finding"
                continue
            x_min = row['x_min']
            y_min = row['y_min']
            x_max = row['x_max']
            y_max = row['y_max']
            if pd.isna(x_min) or pd.isna(y_min) or pd.isna(x_max) or pd.isna(y_max):
                continue
            # Scale to resized image
            x_min = max(0, x_min * scale_x)
            y_min = max(0, y_min * scale_y)
            x_max = min(IMG_SIZE, x_max * scale_x)
            y_max = min(IMG_SIZE, y_max * scale_y)
            if x_max <= x_min or y_max <= y_min:
                continue
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(class_id + 1)  # background is 0

        # Convert boxes and labels to tensors, ensuring correct shape when empty
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        # Data augmentation
        if self.transforms:
            img, boxes = self.apply_augmentation(img, boxes)

        # Ensure array is contiguous before converting to tensor
        img = np.ascontiguousarray(img)

        # Convert to tensor and channel first
        img = torch.from_numpy(img).permute(2, 0, 1).float()

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx]),
        }
        return img, target

    def apply_augmentation(self, img, boxes):
        # Random horizontal flip
        if random.random() < 0.5:
            img = img[:, ::-1, :]
            if boxes.numel() > 0:
                # Flip boxes: x_min' = width - x_max, x_max' = width - x_min
                boxes[:, [0, 2]] = IMG_SIZE - boxes[:, [2, 0]]
        # Random brightness/contrast
        if random.random() < 0.5:
            alpha = random.uniform(0.8, 1.2)   # contrast
            beta = random.uniform(-0.2, 0.2)  # brightness
            img = img * alpha + beta
            img = np.clip(img, 0.0, 1.0)
        return img, boxes

def collate_fn(batch):
    return tuple(zip(*batch))

# mAP computation (PASCAL VOC 2010 style, 11-point interpolation)
def compute_map(predictions, ground_truths, iou_thresh=IOU_THRESH):
    """
    predictions: list of dicts per image, each with keys 'boxes', 'labels', 'scores'
    ground_truths: list of dicts per image, each with keys 'boxes', 'labels'
    Both boxes are assumed to be in resized coordinates (IMG_SIZE).
    Labels: 1..14 (abnormalities)
    """
    # Group by class
    gt_by_class = {c: [] for c in range(1, 15)}   # list of (img_idx, box)
    preds_by_class = {c: [] for c in range(1, 15)} # list of (img_idx, box, score)

    # Populate ground truths
    for img_idx, gt in enumerate(ground_truths):
        boxes = gt['boxes'].cpu().numpy()
        labels = gt['labels'].cpu().numpy()
        for box, label in zip(boxes, labels):
            if label in gt_by_class:
                gt_by_class[label].append((img_idx, box))

    # Populate predictions
    for img_idx, pred in enumerate(predictions):
        boxes = pred['boxes'].cpu().numpy()
        labels = pred['labels'].cpu().numpy()
        scores = pred['scores'].cpu().numpy()
        for box, label, score in zip(boxes, labels, scores):
            if label in preds_by_class:
                preds_by_class[label].append((img_idx, box, score))

    aps = []
    for c in range(1, 15):
        gt_c = gt_by_class[c]
        if len(gt_c) == 0:
            # No ground truth for this class; skip (do not affect mAP)
            continue
        preds_c = preds_by_class[c]
        # Sort predictions by score descending
        preds_c.sort(key=lambda x: x[2], reverse=True)

        # For each image, keep list of unmatched GT boxes
        gt_dict = {}
        for img_idx, box in gt_c:
            gt_dict.setdefault(img_idx, []).append(box)

        # For each image, a boolean list indicating whether GT is used
        matched = {img_idx: [False]*len(gt_dict[img_idx]) for img_idx in gt_dict}

        tp = []
        fp = []
        for img_idx, pred_box, _ in preds_c:
            if img_idx not in gt_dict:
                # No GT in this image -> false positive
                fp.append(1)
                tp.append(0)
                continue
            # Find best IoU with unmatched GT
            best_iou = 0.0
            best_idx = -1
            for i, gt_box in enumerate(gt_dict[img_idx]):
                if matched[img_idx][i]:
                    continue
                iou = compute_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
            if best_iou > iou_thresh:
                tp.append(1)
                fp.append(0)
                matched[img_idx][best_idx] = True
            else:
                tp.append(0)
                fp.append(1)

        tp = np.array(tp)
        fp = np.array(fp)
        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        precisions = cum_tp / (cum_tp + cum_fp + 1e-12)
        recalls = cum_tp / len(gt_c)

        # 11-point interpolation
        ap = 0.0
        for t in np.arange(0, 1.1, 0.1):
            mask = recalls >= t
            if mask.any():
                p = np.max(precisions[mask])
            else:
                p = 0.0
            ap += p / 11.0
        aps.append(ap)

    if len(aps) == 0:
        return 0.0
    return np.mean(aps)

# Main
if __name__ == "__main__":
    print("Loading metadata and preprocessing DICOM...")
    meta = load_meta()

    print("Reading train CSV...")
    df = pd.read_csv(TRAIN_CSV)

    # Get unique image IDs and split
    all_image_ids = df['image_id'].unique()
    np.random.seed(SEED)
    np.random.shuffle(all_image_ids)
    split_idx = int(len(all_image_ids) * (1 - VAL_SPLIT))
    train_ids = all_image_ids[:split_idx]
    val_ids = all_image_ids[split_idx:]

    train_df = df[df['image_id'].isin(train_ids)]
    val_df = df[df['image_id'].isin(val_ids)]

    print(f"Training images: {len(train_ids)}, Validation images: {len(val_ids)}")

    # Datasets
    train_dataset = VinBigDataDataset(train_df, TRAIN_JPG_DIR, meta, transforms=True)
    val_dataset = VinBigDataDataset(val_df, TRAIN_JPG_DIR, meta, transforms=False)

    # DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, collate_fn=collate_fn, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, collate_fn=collate_fn, pin_memory=True
    )

    # Model
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 15)  # 14 abnormalities + background

    model.to(device)

    # Optimizer and scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=5e-5)

    # Training loop
    best_map = 0.0
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for images, targets in progress_bar:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()
            progress_bar.set_postfix(loss=losses.item())

        scheduler.step()
        print(f"Epoch {epoch+1} average loss: {epoch_loss / len(train_loader):.4f}")

        # Validation mAP in last two epochs
        if epoch >= NUM_EPOCHS - 2:
            model.eval()
            all_preds = []
            all_gts = []
            with torch.no_grad():
                for images, targets in tqdm(val_loader, desc="Validating"):
                    images = [img.to(device) for img in images]
                    outputs = model(images)

                    # Move outputs to CPU
                    outputs = [{k: v.cpu() for k, v in o.items()} for o in outputs]
                    all_preds.extend(outputs)
                    all_gts.extend([{k: v.cpu() for k, v in t.items()} for t in targets])

            mAP = compute_map(all_preds, all_gts, iou_thresh=IOU_THRESH)
            print(f"Validation mAP (IoU > {IOU_THRESH}): {mAP:.4f}")

    print(f"Final validation mAP: {mAP:.4f}")

    # --- Inference on test set and submission ---
    print("Generating test predictions...")
    test_image_ids = [f.split('.')[0] for f in os.listdir(TEST_JPG_DIR) if f.endswith('.jpg')]
    if len(test_image_ids) == 0:
        test_image_ids = [k for k in meta.keys() if k in [f.split('.')[0] for f in os.listdir(TEST_DICOM_DIR)]]
    test_image_ids.sort()

    model.eval()
    submission_data = []
    with torch.no_grad():
        for img_id in tqdm(test_image_ids, desc="Test images"):
            img_path = os.path.join(TEST_JPG_DIR, f"{img_id}.jpg")
            if not os.path.exists(img_path):
                img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
            else:
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.astype(np.float32) / 255.0
            img = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).to(device)

            output = model(img)[0]  # batch size 1

            # Filter by confidence
            keep = output['scores'] >= CONF_THRESH
            boxes = output['boxes'][keep].cpu().numpy()
            scores = output['scores'][keep].cpu().numpy()
            labels = output['labels'][keep].cpu().numpy()

            # Scale boxes back to original dimensions
            orig_h, orig_w = meta.get(img_id, (IMG_SIZE, IMG_SIZE))
            scale_x = orig_w / IMG_SIZE
            scale_y = orig_h / IMG_SIZE

            pred_strings = []
            for box, score, label in zip(boxes, scores, labels):
                class_id = label - 1
                x_min = int(max(0, box[0] * scale_x))
                y_min = int(max(0, box[1] * scale_y))
                x_max = int(min(orig_w, box[2] * scale_x))
                y_max = int(min(orig_h, box[3] * scale_y))
                if x_max <= x_min or y_max <= y_min:
                    continue
                pred_strings.append(f"{class_id} {score:.5f} {x_min} {y_min} {x_max} {y_max}")

            if len(pred_strings) == 0:
                pred_str = "14 1.0 0 0 1 1"
            else:
                pred_str = " ".join(pred_strings)

            submission_data.append([img_id, pred_str])

    # Write submission file
    sub_df = pd.DataFrame(submission_data, columns=["image_id", "PredictionString"])
    sub_df.to_csv(SUBMISSION_FILE, index=False)
    print(f"Submission saved to {SUBMISSION_FILE}")

    print("Done.")