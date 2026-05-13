import os
import json
import glob
import numpy as np
import pandas as pd
import pydicom
from pydicom.pixel_data_handlers import apply_voi_lut
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
import timm
from tqdm import tqdm
import warnings
import gc

warnings.filterwarnings("ignore")

# Configuration - REDUCED FOR MEMORY
SEED = 42
IMG_SIZE = 256  # Reduced from 512
BATCH_SIZE = 4  # Reduced from 16
GRAD_ACCUM_STEPS = 4  # For effective batch size of 16
EPOCHS_STUDY = 3  # Reduced from 10
EPOCHS_DET = 3  # Reduced from 8
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4
DATA_DIR = "./input"
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test")
STUDY_CSV = os.path.join(DATA_DIR, "train_study_level.csv")
IMAGE_CSV = os.path.join(DATA_DIR, "train_image_level.csv")
SAMPLE_SUB = os.path.join(DATA_DIR, "sample_submission.csv")

# Set seeds
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Clear GPU memory
torch.cuda.empty_cache()
gc.collect()

# Load data
study_df = pd.read_csv(STUDY_CSV)
image_df = pd.read_csv(IMAGE_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB)

# Preprocess study IDs
study_df["StudyInstanceUID"] = study_df["id"].apply(lambda x: x.replace("_study", ""))
image_df["image_id"] = image_df["id"].apply(lambda x: x.replace("_image", ""))
image_df["StudyInstanceUID"] = image_df["StudyInstanceUID"].astype(str)

# Merge study labels into image dataframe
image_df = image_df.merge(
    study_df[
        [
            "StudyInstanceUID",
            "Negative for Pneumonia",
            "Typical Appearance",
            "Indeterminate Appearance",
            "Atypical Appearance",
        ]
    ],
    on="StudyInstanceUID",
    how="left",
)

# Split by study for validation
studies = study_df["StudyInstanceUID"].unique()
train_studies, val_studies = train_test_split(studies, test_size=0.2, random_state=SEED)
train_image_df = image_df[image_df["StudyInstanceUID"].isin(train_studies)].reset_index(
    drop=True
)
val_image_df = image_df[image_df["StudyInstanceUID"].isin(val_studies)].reset_index(
    drop=True
)

# Study-level labels
study_labels = [
    "Negative for Pneumonia",
    "Typical Appearance",
    "Indeterminate Appearance",
    "Atypical Appearance",
]


# Helper functions
def dicom_to_array(path, voi_lut=True):
    """Read DICOM and convert to normalized array"""
    dicom = pydicom.dcmread(path)
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array
    if dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data
    data = data - np.min(data)
    data = data / (np.max(data) + 1e-6)
    data = (data * 255).astype(np.uint8)
    return data


def resize_and_pad(img, target_size=IMG_SIZE):
    """Resize and pad image to square"""
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    pad_h = target_size - new_h
    pad_w = target_size - new_w
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    img_padded = np.pad(
        img_resized, ((top, bottom), (left, right)), mode="constant", constant_values=0
    )
    return img_padded


def parse_boxes(box_str):
    """Parse bounding box string from CSV"""
    if pd.isna(box_str):
        return []
    try:
        boxes = eval(box_str)
        if isinstance(boxes, dict):
            boxes = [boxes]
        return [
            {"x": b["x"], "y": b["y"], "width": b["width"], "height": b["height"]}
            for b in boxes
            if "x" in b
        ]
    except:
        return []


# Datasets
class StudyClassificationDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, is_train=True):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.is_train = is_train
        self.image_paths = []
        self.study_labels = []
        self.study_ids = []

        # Sample fewer images per study for memory
        for _, row in df.iterrows():
            study_id = row["StudyInstanceUID"]
            label = row[study_labels].values.astype(np.float32)
            # Find all images for this study
            pattern = os.path.join(img_dir, study_id, "*", "*.dcm")
            paths = glob.glob(pattern)
            # Limit to max 2 images per study for training
            if is_train and len(paths) > 2:
                paths = np.random.choice(paths, 2, replace=False)
            elif len(paths) > 4:  # Limit for validation too
                paths = np.random.choice(paths, 4, replace=False)
            for path in paths:
                self.image_paths.append(path)
                self.study_labels.append(label)
                self.study_ids.append(study_id)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = dicom_to_array(img_path)
        img = resize_and_pad(img, IMG_SIZE)
        img = np.stack([img, img, img], axis=-1)  # Convert to 3-channel

        if self.transform:
            img = self.transform(img)

        label = torch.tensor(self.study_labels[idx])
        study_id = self.study_ids[idx]
        return img, label, study_id


class DetectionDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, is_train=True):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row["image_id"]
        study_id = row["StudyInstanceUID"]

        # Find image path
        pattern = os.path.join(self.img_dir, study_id, "*", f"{image_id}.dcm")
        paths = glob.glob(pattern)
        img_path = paths[0] if paths else None
        if img_path is None:
            # Try alternative pattern
            pattern = os.path.join(self.img_dir, "**", f"{image_id}.dcm")
            paths = glob.glob(pattern, recursive=True)
            img_path = paths[0] if paths else None

        img = dicom_to_array(img_path)
        img = resize_and_pad(img, IMG_SIZE)
        img = np.stack([img, img, img], axis=-1)

        # Parse boxes
        boxes = parse_boxes(row["boxes"])
        if boxes and self.is_train:
            # Convert to xmin, ymin, xmax, ymax
            h, w = img.shape[:2]
            scale = IMG_SIZE / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)

            bbox_list = []
            for box in boxes:
                x = box["x"] * scale
                y = box["y"] * scale
                width = box["width"] * scale
                height = box["height"] * scale
                xmin = x
                ymin = y
                xmax = x + width
                ymax = y + height
                # Clip to image bounds
                xmin = max(0, min(xmin, new_w - 1))
                ymin = max(0, min(ymin, new_h - 1))
                xmax = max(0, min(xmax, new_w))
                ymax = max(0, min(ymax, new_h))
                if xmax > xmin and ymax > ymin:
                    bbox_list.append([xmin, ymin, xmax, ymax])

            if bbox_list:
                boxes_tensor = torch.tensor(bbox_list, dtype=torch.float32)
                labels_tensor = torch.ones((len(bbox_list),), dtype=torch.int64)
            else:
                boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
                labels_tensor = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)

        if self.transform:
            img = self.transform(img)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([idx]),
            "area": (
                (boxes_tensor[:, 3] - boxes_tensor[:, 1])
                * (boxes_tensor[:, 2] - boxes_tensor[:, 0])
                if len(boxes_tensor) > 0
                else torch.tensor([])
            ),
            "iscrowd": (
                torch.zeros((len(boxes_tensor),), dtype=torch.int64)
                if len(boxes_tensor) > 0
                else torch.tensor([])
            ),
        }

        return img, target, image_id


# Transforms
train_transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

val_transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Create datasets and dataloaders
study_train_ds = StudyClassificationDataset(
    train_image_df, TRAIN_IMG_DIR, train_transform, is_train=True
)
study_val_ds = StudyClassificationDataset(
    val_image_df, TRAIN_IMG_DIR, val_transform, is_train=False
)
det_train_ds = DetectionDataset(
    train_image_df, TRAIN_IMG_DIR, train_transform, is_train=True
)
det_val_ds = DetectionDataset(
    val_image_df, TRAIN_IMG_DIR, val_transform, is_train=False
)

study_train_loader = DataLoader(
    study_train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)
study_val_loader = DataLoader(
    study_val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)
det_train_loader = DataLoader(
    det_train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    collate_fn=lambda x: tuple(zip(*x)),
)
det_val_loader = DataLoader(
    det_val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    collate_fn=lambda x: tuple(zip(*x)),
)


# Study classification model - LIGHTER MODEL
class StudyModel(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        # Use smaller model
        self.backbone = timm.create_model(
            "efficientnet_b0", pretrained=True, num_classes=0
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1280, num_classes)

    def forward(self, x):
        features = self.backbone.forward_features(x)
        pooled = self.pool(features).flatten(1)
        return self.fc(pooled)


study_model = StudyModel().to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(study_model.parameters(), lr=LR)

# Train study model with gradient accumulation
print("Training study classification model...")
for epoch in range(EPOCHS_STUDY):
    study_model.train()
    running_loss = 0.0
    optimizer.zero_grad()

    for batch_idx, (images, labels, _) in enumerate(
        tqdm(study_train_loader, desc=f"Study Epoch {epoch+1}/{EPOCHS_STUDY}")
    ):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = study_model(images)
        loss = criterion(outputs, labels) / GRAD_ACCUM_STEPS
        loss.backward()

        if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()

        running_loss += loss.item() * GRAD_ACCUM_STEPS

    # Validation
    study_model.eval()
    val_preds = []
    val_labels = []
    with torch.no_grad():
        for images, labels, _ in study_val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = study_model(images)
            val_preds.append(torch.sigmoid(outputs).cpu().numpy())
            val_labels.append(labels.cpu().numpy())

    val_preds = np.vstack(val_preds)
    val_labels = np.vstack(val_labels)
    ap_scores = []
    for i in range(4):
        ap = average_precision_score(val_labels[:, i], val_preds[:, i])
        ap_scores.append(ap)
    mean_ap = np.mean(ap_scores)
    print(
        f"Study Epoch {epoch+1}: Loss = {running_loss/len(study_train_loader):.4f}, mAP = {mean_ap:.4f}"
    )

# Detection model
detection_model = fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 2  # background and opacity
in_features = detection_model.roi_heads.box_predictor.cls_score.in_features
detection_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
detection_model = detection_model.to(DEVICE)

params = [p for p in detection_model.parameters() if p.requires_grad]
optimizer = optim.Adam(params, lr=LR)

# Train detection model with gradient accumulation
print("\nTraining detection model...")
for epoch in range(EPOCHS_DET):
    detection_model.train()
    running_loss = 0.0
    optimizer.zero_grad()

    for batch_idx, (images, targets, _) in enumerate(
        tqdm(det_train_loader, desc=f"Det Epoch {epoch+1}/{EPOCHS_DET}")
    ):
        images = [img.to(DEVICE) for img in images]
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        loss_dict = detection_model(images, targets)
        losses = sum(loss for loss in loss_dict.values()) / GRAD_ACCUM_STEPS
        losses.backward()

        if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()

        running_loss += losses.item() * GRAD_ACCUM_STEPS

    print(f"Det Epoch {epoch+1}: Loss = {running_loss/len(det_train_loader):.4f}")

# Validation evaluation
print("\nEvaluating on validation set...")
study_model.eval()
detection_model.eval()

# Study predictions
study_pred_dict = {}
study_true_dict = {}
with torch.no_grad():
    for images, labels, study_ids in study_val_loader:
        images = images.to(DEVICE)
        outputs = study_model(images)
        probs = torch.sigmoid(outputs).cpu().numpy()

        for i, study_id in enumerate(study_ids):
            if study_id not in study_pred_dict:
                study_pred_dict[study_id] = []
                study_true_dict[study_id] = labels[i].numpy()
            study_pred_dict[study_id].append(probs[i])

# Average predictions per study
for study_id in study_pred_dict:
    study_pred_dict[study_id] = np.mean(study_pred_dict[study_id], axis=0)

# Detection predictions
det_pred_dict = {}
det_true_dict = {}
with torch.no_grad():
    for images, targets, image_ids in det_val_loader:
        images = [img.to(DEVICE) for img in images]
        outputs = detection_model(images)

        for i, image_id in enumerate(image_ids):
            pred_boxes = outputs[i]["boxes"].cpu().numpy()
            pred_scores = outputs[i]["scores"].cpu().numpy()
            pred_labels = outputs[i]["labels"].cpu().numpy()

            # Filter by confidence
            mask = pred_scores > 0.3  # Lower threshold
            pred_boxes = pred_boxes[mask]
            pred_scores = pred_scores[mask]
            pred_labels = pred_labels[mask]

            det_pred_dict[image_id] = {
                "boxes": pred_boxes,
                "scores": pred_scores,
                "labels": pred_labels,
            }

            # Ground truth
            gt_boxes = targets[i]["boxes"].cpu().numpy()
            gt_labels = targets[i]["labels"].cpu().numpy()
            det_true_dict[image_id] = {"boxes": gt_boxes, "labels": gt_labels}


# Convert to VOC format for mAP calculation
def convert_to_voc(pred_dict, true_dict, is_study=False):
    """Convert predictions to VOC format for mAP calculation"""
    all_preds = []
    all_gts = []

    for obj_id in pred_dict:
        if is_study:
            # Study-level: 4 classes with dummy boxes
            pred_probs = pred_dict[obj_id]
            gt_labels = true_dict[obj_id]

            for class_idx in range(4):
                # Prediction
                if pred_probs[class_idx] > 0.1:  # Low threshold to include all
                    all_preds.append(
                        {
                            "image_id": f"{obj_id}_study",
                            "class_id": class_idx,
                            "bbox": [0, 0, 1, 1],  # Dummy box
                            "score": float(pred_probs[class_idx]),
                        }
                    )

                # Ground truth
                if gt_labels[class_idx] == 1:
                    all_gts.append(
                        {
                            "image_id": f"{obj_id}_study",
                            "class_id": class_idx,
                            "bbox": [0, 0, 1, 1],
                            "difficult": 0,
                        }
                    )
        else:
            # Image-level: opacity class
            pred_data = pred_dict[obj_id]
            gt_data = true_dict[obj_id]

            # Predictions
            for box, score, label in zip(
                pred_data["boxes"], pred_data["scores"], pred_data["labels"]
            ):
                if label == 1:  # opacity class
                    xmin, ymin, xmax, ymax = box
                    all_preds.append(
                        {
                            "image_id": f"{obj_id}_image",
                            "class_id": 4,  # opacity is class 4
                            "bbox": [xmin, ymin, xmax, ymax],
                            "score": float(score),
                        }
                    )

            # If no predictions, add "none"
            if len(pred_data["boxes"]) == 0:
                all_preds.append(
                    {
                        "image_id": f"{obj_id}_image",
                        "class_id": 5,  # none is class 5
                        "bbox": [0, 0, 1, 1],
                        "score": 1.0,
                    }
                )

            # Ground truth
            if len(gt_data["boxes"]) > 0:
                for box, label in zip(gt_data["boxes"], gt_data["labels"]):
                    if label == 1:
                        xmin, ymin, xmax, ymax = box
                        all_gts.append(
                            {
                                "image_id": f"{obj_id}_image",
                                "class_id": 4,
                                "bbox": [xmin, ymin, xmax, ymax],
                                "difficult": 0,
                            }
                        )
            else:
                all_gts.append(
                    {
                        "image_id": f"{obj_id}_image",
                        "class_id": 5,
                        "bbox": [0, 0, 1, 1],
                        "difficult": 0,
                    }
                )

    return all_preds, all_gts


# Combine study and image predictions
study_preds, study_gts = convert_to_voc(study_pred_dict, study_true_dict, is_study=True)
det_preds, det_gts = convert_to_voc(det_pred_dict, det_true_dict, is_study=False)
all_preds = study_preds + det_preds
all_gts = study_gts + det_gts


# Calculate mAP using Pascal VOC 2010 style
def calculate_map(preds, gts, iou_threshold=0.5):
    """Calculate mean Average Precision in Pascal VOC 2010 style"""
    # Group by class
    class_ids = set([p["class_id"] for p in preds] + [g["class_id"] for g in gts])
    aps = []

    for class_id in class_ids:
        # Get predictions and ground truths for this class
        class_preds = [p for p in preds if p["class_id"] == class_id]
        class_gts = [g for g in gts if g["class_id"] == class_id]

        # Sort predictions by confidence
        class_preds.sort(key=lambda x: x["score"], reverse=True)

        # Group ground truths by image
        gt_by_image = {}
        for gt in class_gts:
            if gt["image_id"] not in gt_by_image:
                gt_by_image[gt["image_id"]] = []
            gt_by_image[gt["image_id"]].append(gt)

        # Initialize
        tp = np.zeros(len(class_preds))
        fp = np.zeros(len(class_preds))
        n_positives = len(class_gts)

        # Match predictions to ground truths
        gt_matched = {img_id: [False] * len(gts) for img_id, gts in gt_by_image.items()}

        for i, pred in enumerate(class_preds):
            img_id = pred["image_id"]
            pred_bbox = pred["bbox"]

            if img_id not in gt_by_image:
                fp[i] = 1
                continue

            # Find best matching ground truth
            best_iou = 0
            best_idx = -1
            for j, gt in enumerate(gt_by_image[img_id]):
                gt_bbox = gt["bbox"]
                # Calculate IoU
                x1 = max(pred_bbox[0], gt_bbox[0])
                y1 = max(pred_bbox[1], gt_bbox[1])
                x2 = min(pred_bbox[2], gt_bbox[2])
                y2 = min(pred_bbox[3], gt_bbox[3])

                if x2 <= x1 or y2 <= y1:
                    iou = 0
                else:
                    intersection = (x2 - x1) * (y2 - y1)
                    pred_area = (pred_bbox[2] - pred_bbox[0]) * (
                        pred_bbox[3] - pred_bbox[1]
                    )
                    gt_area = (gt_bbox[2] - gt_bbox[0]) * (gt_bbox[3] - gt_bbox[1])
                    union = pred_area + gt_area - intersection
                    iou = intersection / union if union > 0 else 0

                if iou > best_iou:
                    best_iou = iou
                    best_idx = j

            if best_iou >= iou_threshold and not gt_matched[img_id][best_idx]:
                tp[i] = 1
                gt_matched[img_id][best_idx] = True
            else:
                fp[i] = 1

        # Calculate precision-recall
        fp_cumsum = np.cumsum(fp)
        tp_cumsum = np.cumsum(tp)
        recalls = tp_cumsum / (n_positives + 1e-8)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)

        # VOC 2010: 11-point interpolation
        ap = 0
        for t in np.arange(0, 1.1, 0.1):
            mask = recalls >= t
            if mask.any():
                ap += np.max(precisions[mask])
        ap /= 11
        aps.append(ap)

    return np.mean(aps) if aps else 0


mAP = calculate_map(all_preds, all_gts)
print(f"\nValidation mAP (IoU>0.5): {mAP:.4f}")

# Test prediction
print("\nGenerating test predictions...")

# Get test IDs from sample submission
test_study_ids = []
test_image_ids = []
for id_str in sample_sub["id"]:
    if "_study" in id_str:
        test_study_ids.append(id_str.replace("_study", ""))
    elif "_image" in id_str:
        test_image_ids.append(id_str.replace("_image", ""))

# Find all test images
test_image_paths = []
test_image_to_study = {}
for study_id in test_study_ids:
    pattern = os.path.join(TEST_IMG_DIR, study_id, "*", "*.dcm")
    paths = glob.glob(pattern)
    # Limit to max 2 images per study for memory
    if len(paths) > 2:
        paths = paths[:2]
    for path in paths:
        image_id = os.path.splitext(os.path.basename(path))[0]
        test_image_paths.append(path)
        test_image_to_study[image_id] = study_id

# Additional test images from sample submission
for image_id in test_image_ids:
    if image_id not in test_image_to_study:
        pattern = os.path.join(TEST_IMG_DIR, "**", f"{image_id}.dcm")
        paths = glob.glob(pattern, recursive=True)
        if paths:
            test_image_paths.append(paths[0])
            # Try to extract study_id from path
            path_parts = paths[0].split(os.sep)
            if len(path_parts) >= 4:
                test_image_to_study[image_id] = path_parts[-3]


# Prepare test data loader
class TestDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image_id = os.path.splitext(os.path.basename(img_path))[0]
        img = dicom_to_array(img_path)
        img = resize_and_pad(img, IMG_SIZE)
        img = np.stack([img, img, img], axis=-1)

        if self.transform:
            img = self.transform(img)

        return img, image_id, img_path


test_dataset = TestDataset(test_image_paths, val_transform)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

# Predict study labels
study_model.eval()
study_preds_test = {}
with torch.no_grad():
    for images, image_ids, _ in tqdm(test_loader, desc="Predicting study labels"):
        images = images.to(DEVICE)
        outputs = study_model(images)
        probs = torch.sigmoid(outputs).cpu().numpy()

        for i, image_id in enumerate(image_ids):
            study_id = test_image_to_study.get(image_id, None)
            if study_id:
                if study_id not in study_preds_test:
                    study_preds_test[study_id] = []
                study_preds_test[study_id].append(probs[i])

# Average predictions per study
final_study_preds = {}
for study_id in study_preds_test:
    avg_probs = np.mean(study_preds_test[study_id], axis=0)
    final_study_preds[study_id] = avg_probs

# Predict detections
detection_model.eval()
det_preds_test = {}
with torch.no_grad():
    for images, image_ids, img_paths in tqdm(test_loader, desc="Predicting detections"):
        images = [img.to(DEVICE) for img in images]
        outputs = detection_model(images)

        for i, image_id in enumerate(image_ids):
            pred_boxes = outputs[i]["boxes"].cpu().numpy()
            pred_scores = outputs[i]["scores"].cpu().numpy()
            pred_labels = outputs[i]["labels"].cpu().numpy()

            # Filter by confidence
            mask = pred_scores > 0.3  # Lower threshold
            pred_boxes = pred_boxes[mask]
            pred_scores = pred_scores[mask]
            pred_labels = pred_labels[mask]

            det_preds_test[image_id] = {
                "boxes": pred_boxes,
                "scores": pred_scores,
                "labels": pred_labels,
            }

# Create submission
print("\nCreating submission file...")
submission_rows = []

# Study predictions
for study_id in test_study_ids:
    if study_id in final_study_preds:
        probs = final_study_preds[study_id]
        pred_strings = []
        for class_idx, class_name in enumerate(
            ["negative", "typical", "indeterminate", "atypical"]
        ):
            if probs[class_idx] > 0.1:  # Include if probability > 0.1
                pred_strings.append(f"{class_name} {probs[class_idx]:.4f} 0 0 1 1")

        # If no predictions, add the highest probability one
        if not pred_strings:
            max_idx = np.argmax(probs)
            class_name = ["negative", "typical", "indeterminate", "atypical"][max_idx]
            pred_strings.append(f"{class_name} {probs[max_idx]:.4f} 0 0 1 1")

        submission_rows.append(
            {"id": f"{study_id}_study", "PredictionString": " ".join(pred_strings)}
        )
    else:
        # Default prediction if study not found
        submission_rows.append(
            {"id": f"{study_id}_study", "PredictionString": "negative 1.0 0 0 1 1"}
        )

# Image predictions
for image_id in test_image_ids:
    if image_id in det_preds_test:
        pred_data = det_preds_test[image_id]
        pred_strings = []

        if len(pred_data["boxes"]) > 0:
            for box, score in zip(pred_data["boxes"], pred_data["scores"]):
                xmin, ymin, xmax, ymax = box
                # Convert to integer coordinates
                xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
                pred_strings.append(f"opacity {score:.4f} {xmin} {ymin} {xmax} {ymax}")
        else:
            pred_strings.append("none 1.0 0 0 1 1")

        submission_rows.append(
            {"id": f"{image_id}_image", "PredictionString": " ".join(pred_strings)}
        )
    else:
        # Default prediction if image not found
        submission_rows.append(
            {"id": f"{image_id}_image", "PredictionString": "none 1.0 0 0 1 1"}
        )

# Create submission DataFrame
submission_df = pd.DataFrame(submission_rows)

# Ensure we have all IDs from sample submission
missing_ids = set(sample_sub["id"]) - set(submission_df["id"])
for missing_id in missing_ids:
    if "_study" in missing_id:
        submission_df = pd.concat(
            [
                submission_df,
                pd.DataFrame(
                    [{"id": missing_id, "PredictionString": "negative 1.0 0 0 1 1"}]
                ),
            ],
            ignore_index=True,
        )
    elif "_image" in missing_id:
        submission_df = pd.concat(
            [
                submission_df,
                pd.DataFrame(
                    [{"id": missing_id, "PredictionString": "none 1.0 0 0 1 1"}]
                ),
            ],
            ignore_index=True,
        )

# Sort by ID to match sample submission order
submission_df = submission_df.sort_values("id").reset_index(drop=True)

# Save submission
os.makedirs("./submission", exist_ok=True)
submission_path = "./submission/submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"\nSubmission saved to {submission_path}")
print(f"Shape: {submission_df.shape}")
print(f"Validation mAP: {mAP:.4f}")

# Also save to working directory for backup
os.makedirs("./working", exist_ok=True)
backup_path = "./working/submission.csv"
submission_df.to_csv(backup_path, index=False)
print(f"Backup saved to {backup_path}")
