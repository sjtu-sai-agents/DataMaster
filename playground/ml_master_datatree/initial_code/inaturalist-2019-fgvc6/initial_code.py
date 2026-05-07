import os
import json
import random
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True   # for speed

# Paths
INPUT_DIR = "./input"
TRAIN_VAL_DIR = os.path.join(INPUT_DIR, "train_val2019")
TEST_DIR = os.path.join(INPUT_DIR, "test2019")
TEST_JSON = os.path.join(INPUT_DIR, "test2019.json")
SUBMISSION_PATH = "./submission/submission.csv"
os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----------------------------------------------------------------------
# Collect training images and labels from directory structure
print("Collecting training images...")
train_image_paths = []
train_labels_raw = []
for root, dirs, files in os.walk(TRAIN_VAL_DIR):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg')):
            path = os.path.join(root, file)
            label_str = os.path.basename(root)   # class id folder
            if label_str.isdigit():
                train_image_paths.append(path)
                train_labels_raw.append(int(label_str))

# Encode labels to 0..N-1
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(train_labels_raw)
num_classes = len(label_encoder.classes_)
print(f"Number of classes: {num_classes}")

# Train/validation split (stratified)
train_paths, val_paths, train_enc, val_enc = train_test_split(
    train_image_paths, labels_encoded,
    test_size=0.1, stratify=labels_encoded, random_state=SEED
)
print(f"Training samples: {len(train_paths)}, Validation samples: {len(val_paths)}")

# ----------------------------------------------------------------------
# Transforms
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ----------------------------------------------------------------------
# Dataset class
class INatDataset(Dataset):
    def __init__(self, paths, labels, transform=None, is_test=False):
        self.paths = paths
        self.labels = labels
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        if self.transform:
            image = self.transform(image)
        if self.is_test:
            return image, self.labels[idx]   # label is image id
        else:
            return image, self.labels[idx]

# ----------------------------------------------------------------------
# DataLoaders
batch_size = 128
num_workers = 8  # use multiple workers for faster loading

train_dataset = INatDataset(train_paths, train_enc, transform=train_transform)
val_dataset = INatDataset(val_paths, val_enc, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)

# ----------------------------------------------------------------------
# Model
def get_model(num_classes):
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model

model = get_model(num_classes).to(device)
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
steps_per_epoch = len(train_loader)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=0.001, epochs=5, steps_per_epoch=steps_per_epoch
)
scaler = torch.cuda.amp.GradScaler()

# ----------------------------------------------------------------------
# Training loop
best_val_acc = 0.0
best_model_path = "best_model.pth"

for epoch in range(5):
    # Training phase
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/5 [Train]")
    for inputs, targets in loop:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        train_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        loop.set_postfix(loss=loss.item(), acc=100.*correct/total)

    train_acc = 100. * correct / total

    # Validation phase
    model.eval()
    val_correct = 0
    val_total = 0
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/5 [Val]"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            val_total += targets.size(0)
            val_correct += predicted.eq(targets).sum().item()
    val_acc = 100. * val_correct / val_total
    print(f"Epoch {epoch+1}: Train Acc = {train_acc:.2f}%, Val Acc = {val_acc:.2f}%")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_model_path)
        print(f"  -> New best model saved (val acc {val_acc:.2f}%)")

# ----------------------------------------------------------------------
# Load best model and compute final validation error
model.load_state_dict(torch.load(best_model_path))
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, targets in val_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

val_accuracy = correct / total
val_error = 1.0 - val_accuracy
print(f"\nValidation top-1 error: {val_error:.4f}")

# ----------------------------------------------------------------------
# Prepare test data
print("\nPreparing test data...")
test_paths = []
test_ids = []
if os.path.exists(TEST_JSON):
    with open(TEST_JSON, 'r') as f:
        data = json.load(f)
    for img in data["images"]:
        img_id = img["id"]
        fname = img["file_name"]
        if fname.startswith("test2019/"):
            full_path = os.path.join(INPUT_DIR, fname)
        else:
            full_path = os.path.join(TEST_DIR, fname)
        test_ids.append(img_id)
        test_paths.append(full_path)
    print(f"Loaded {len(test_ids)} test images from JSON.")
else:
    # fallback: scan test directory (should not happen)
    for fname in os.listdir(TEST_DIR):
        if fname.lower().endswith(('.jpg', '.jpeg')):
            test_paths.append(os.path.join(TEST_DIR, fname))
            test_ids.append(os.path.splitext(fname)[0])
    print(f"Loaded {len(test_ids)} test images from directory.")

test_dataset = INatDataset(test_paths, test_ids, transform=val_transform, is_test=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, pin_memory=True)

# ----------------------------------------------------------------------
# Generate predictions (top-5)
model.eval()
all_ids = []
all_preds = []

with torch.no_grad():
    for inputs, ids in tqdm(test_loader, desc="Predicting"):
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, top5 = outputs.topk(5, dim=1)   # shape (batch,5)
        # Convert indices to original class IDs
        batch_preds = label_encoder.inverse_transform(top5.cpu().numpy().reshape(-1))
        batch_preds = batch_preds.reshape(-1, 5)
        all_ids.extend(ids.cpu().numpy().tolist())
        for row in batch_preds:
            all_preds.append(' '.join(str(x) for x in row))

# Create submission DataFrame
sub_df = pd.DataFrame({"id": all_ids, "predicted": all_preds})
sub_df.to_csv(SUBMISSION_PATH, index=False)
print(f"Submission saved to {SUBMISSION_PATH}")