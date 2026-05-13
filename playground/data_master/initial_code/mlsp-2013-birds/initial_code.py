import os
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
DATA_ROOT = "./input"
ESSENTIAL = os.path.join(DATA_ROOT, "essential_data")
SUPPLEMENTAL = os.path.join(DATA_ROOT, "supplemental_data")
SPECTROGRAMS_DIR = os.path.join(SUPPLEMENTAL, "filtered_spectrograms")

NUM_CLASSES = 19
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4
EPOCHS = 30
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)

# ----------------------------------------------------------------------
# Parse mapping: rec_id -> base filename (without .wav)
# ----------------------------------------------------------------------
mapping = {}
with open(os.path.join(ESSENTIAL, "rec_id2filename.txt"), "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = [p.strip().strip('"') for p in line.split(",")]
        try:
            rec_id = int(parts[0])          # skip header if conversion fails
        except ValueError:
            continue
        wav_name = parts[1]
        base = os.path.splitext(wav_name)[0]   # remove .wav
        mapping[rec_id] = base

# ----------------------------------------------------------------------
# Parse training labels from rec_labels_test_hidden.txt
# ----------------------------------------------------------------------
labeled_items = []      # each element: {'rec_id': int, 'base': str, 'labels': list}
with open(os.path.join(ESSENTIAL, "rec_labels_test_hidden.txt"), "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        if "?" in line:                     # skip test recordings
            continue
        parts = line.split(",")
        try:
            rec_id = int(parts[0])          # skip header if conversion fails
        except ValueError:
            continue
        labels = []
        for s in parts[1:]:
            if s:
                try:
                    labels.append(int(s))
                except ValueError:
                    pass
        if rec_id in mapping:
            labeled_items.append({
                'rec_id': rec_id,
                'base': mapping[rec_id],
                'labels': labels
            })
        else:
            warnings.warn(f"rec_id {rec_id} not in mapping, skipping")

print(f"Number of labeled training recordings: {len(labeled_items)}")
if len(labeled_items) == 0:
    raise RuntimeError("No training items found. Check file paths and parsing.")

# ----------------------------------------------------------------------
# Split into training and validation (80% / 20%)
# ----------------------------------------------------------------------
train_items, val_items = train_test_split(
    labeled_items, test_size=0.2, random_state=RANDOM_SEED
)

# ----------------------------------------------------------------------
# Dataset class
# ----------------------------------------------------------------------
class BirdSpectrogramDataset(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        rec_id = item['rec_id']
        base = item['base']
        labels = item['labels']

        # Load and preprocess spectrogram
        img_path = os.path.join(SPECTROGRAMS_DIR, f"{base}.bmp")
        if os.path.exists(img_path):
            try:
                img = Image.open(img_path).convert('L')   # grayscale

                # Convert to tensor [0,1]
                img_tensor = torch.from_numpy(np.array(img)).float().div(255.0)

                # Per‑image standardization
                mean = img_tensor.mean()
                std = img_tensor.std()
                if std > 1e-6:
                    img_tensor = (img_tensor - mean) / std
                else:
                    img_tensor = img_tensor - mean

                # Expand to 3 channels
                img_tensor = img_tensor.unsqueeze(0)          # (1, H, W)
                img_tensor = img_tensor.repeat(3, 1, 1)       # (3, H, W)

                # Resize
                resize = transforms.Resize((IMG_SIZE, IMG_SIZE), antialias=True)
                img_tensor = resize(img_tensor)

                # Normalize with ImageNet statistics
                normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])
                img_tensor = normalize(img_tensor)

            except Exception as e:
                warnings.warn(f"Failed to load {img_path}: {e}, using zero tensor")
                img_tensor = torch.zeros((3, IMG_SIZE, IMG_SIZE), dtype=torch.float)
        else:
            warnings.warn(f"Spectrogram not found: {img_path}, using zero tensor")
            img_tensor = torch.zeros((3, IMG_SIZE, IMG_SIZE), dtype=torch.float)

        # Multi‑hot label vector
        label_vec = torch.zeros(NUM_CLASSES, dtype=torch.float)
        for s in labels:
            if 0 <= s < NUM_CLASSES:
                label_vec[s] = 1.0

        return img_tensor, label_vec, rec_id

# ----------------------------------------------------------------------
# Create data loaders
# ----------------------------------------------------------------------
train_dataset = BirdSpectrogramDataset(train_items)
val_dataset = BirdSpectrogramDataset(val_items)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

# ----------------------------------------------------------------------
# Model, loss, optimizer, scheduler
# ----------------------------------------------------------------------
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
model = model.to(DEVICE)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# ----------------------------------------------------------------------
# Training loop with validation AUC
# ----------------------------------------------------------------------
best_val_auc = 0.0
best_model_state = None

for epoch in range(EPOCHS):
    # Training
    model.train()
    train_loss = 0.0
    for images, labels, _ in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)
    train_loss /= len(train_loader.dataset)

    # Validation
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels, _ in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    if all_preds:
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        val_auc = roc_auc_score(all_labels.ravel(), all_preds.ravel())
    else:
        val_auc = 0.5

    print(f"Epoch {epoch+1}/{EPOCHS} - Train loss: {train_loss:.4f} - Val AUC: {val_auc:.4f}")

    scheduler.step()

    if val_auc > best_val_auc:
        best_val_auc = val_auc
        best_model_state = model.state_dict().copy()

# Load best model
model.load_state_dict(best_model_state)

# ----------------------------------------------------------------------
# Predict on test recordings (from sample_submission)
# ----------------------------------------------------------------------
sub_sample = pd.read_csv(os.path.join(DATA_ROOT, "sample_submission.csv"))
sub_sample['rec_id'] = sub_sample['Id'].astype(int) // 100
test_rec_ids = sub_sample['rec_id'].unique()
print(f"Number of test recordings: {len(test_rec_ids)}")

# Build test items
test_items = []
for rec_id in test_rec_ids:
    if rec_id in mapping:
        test_items.append({
            'rec_id': rec_id,
            'base': mapping[rec_id],
            'labels': []          # dummy
        })
    else:
        warnings.warn(f"Test rec_id {rec_id} not in mapping, skipping")

test_dataset = BirdSpectrogramDataset(test_items)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

# Inference
model.eval()
pred_dict = {}          # rec_id -> array of 19 probabilities
with torch.no_grad():
    for images, _, rec_ids in test_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        probs = torch.sigmoid(outputs).cpu().numpy()
        for i, rec_id in enumerate(rec_ids):
            rec_id = rec_id.item()
            pred_dict[rec_id] = probs[i, :]

# ----------------------------------------------------------------------
# Create submission.csv
# ----------------------------------------------------------------------
submission_df = sub_sample.copy()
submission_df['Probability'] = 0.0
for idx, row in submission_df.iterrows():
    rec_id = row['rec_id']
    species = int(row['Id']) % 100
    if rec_id in pred_dict:
        submission_df.at[idx, 'Probability'] = pred_dict[rec_id][species]
    else:
        submission_df.at[idx, 'Probability'] = 0.0

# Keep only required columns
submission_df = submission_df[['Id', 'Probability']]

# Save
os.makedirs('submission', exist_ok=True)
submission_path = 'submission/submission.csv'
submission_df.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
print(f"Best validation AUC: {best_val_auc:.4f}")