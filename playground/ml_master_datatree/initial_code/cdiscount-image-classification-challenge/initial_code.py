import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import pandas as pd
import numpy as np
from PIL import Image
import bson
import io
import os
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Configuration - adjusted for memory constraints
BATCH_SIZE = 32
IMAGE_SIZE = 224
NUM_WORKERS = 8
EPOCHS = 3
LR = 1e-4
MAX_TRAIN_SAMPLES = 50000
MAX_VAL_SAMPLES = 10000

# Data paths
INPUT_DIR = "./input"
TRAIN_BSON = os.path.join(INPUT_DIR, "train.bson")
TEST_BSON = os.path.join(INPUT_DIR, "test.bson")
CATEGORY_CSV = os.path.join(INPUT_DIR, "category_names.csv")
SAMPLE_SUB = os.path.join(INPUT_DIR, "sample_submission.csv")

# Create submission directory
os.makedirs("./submission", exist_ok=True)

# Load category data
category_df = pd.read_csv(CATEGORY_CSV)
category_ids = category_df["category_id"].unique()
category_to_label = {cat: idx for idx, cat in enumerate(category_ids)}
label_to_category = {idx: cat for idx, cat in enumerate(category_ids)}
num_classes = len(category_ids)

print(f"Number of classes: {num_classes}")

# Image transformations
transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# Dataset class for BSON data - simplified to use only first image
class ProductDataset(Dataset):
    def __init__(self, bson_path, max_samples=None, is_train=True):
        self.bson_path = bson_path
        self.is_train = is_train
        self.max_samples = max_samples
        self.data = []

        # Read BSON file
        print(f"Loading data from {bson_path}...")
        with open(bson_path, "rb") as f:
            data = bson.decode_all(f.read())

        # Limit samples if specified
        if max_samples and len(data) > max_samples:
            data = data[:max_samples]

        for item in tqdm(data):
            product_id = item["_id"]

            # Get category for train data
            if is_train:
                category_id = item.get("category_id")
                if category_id not in category_to_label:
                    continue
                label = category_to_label[category_id]
            else:
                label = None

            # Process only the first image to save memory
            images = item.get("imgs", [])
            if images:
                try:
                    img_data = images[0]["picture"]
                    self.data.append(
                        {
                            "product_id": product_id,
                            "image_data": img_data,
                            "label": label,
                        }
                    )
                except:
                    continue

        print(f"Loaded {len(self.data)} products")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Load image
        img = Image.open(io.BytesIO(item["image_data"])).convert("RGB")
        img_tensor = transform(img)

        if self.is_train:
            return img_tensor, item["label"], item["product_id"]
        else:
            return img_tensor, item["product_id"]


# Simple ResNet18 model
class SimpleResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)


# Prepare datasets
print("Preparing datasets...")
train_dataset = ProductDataset(TRAIN_BSON, max_samples=MAX_TRAIN_SAMPLES, is_train=True)
val_dataset = ProductDataset(TRAIN_BSON, max_samples=MAX_VAL_SAMPLES, is_train=True)

# Create data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

# Initialize model, loss, optimizer
model = SimpleResNet(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# Training loop
print("Starting training...")
best_val_acc = 0.0

for epoch in range(EPOCHS):
    # Training phase
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - Train")
    for images, labels, _ in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

        pbar.set_postfix(
            {"loss": loss.item(), "acc": 100 * train_correct / train_total}
        )

    train_acc = 100 * train_correct / train_total

    # Validation phase
    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - Val")
        for images, labels, _ in pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

            pbar.set_postfix({"acc": 100 * val_correct / val_total})

    val_acc = 100 * val_correct / val_total

    print(f"Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "./working/best_model.pth")

print(f"Best validation accuracy: {best_val_acc:.2f}%")

# Load best model
model.load_state_dict(torch.load("./working/best_model.pth"))
model.eval()

# Process test set
print("Processing test set...")
test_predictions = []

# Process test data in batches to avoid memory issues
with open(TEST_BSON, "rb") as f:
    data = bson.decode_all(f.read())

batch_size = 32
for i in tqdm(range(0, len(data), batch_size), desc="Processing test data"):
    batch_data = data[i : i + batch_size]

    batch_images = []
    batch_ids = []

    for item in batch_data:
        product_id = item["_id"]
        images = item.get("imgs", [])

        if images:
            try:
                img_data = images[0]["picture"]
                img = Image.open(io.BytesIO(img_data)).convert("RGB")
                img_tensor = transform(img)
                batch_images.append(img_tensor)
                batch_ids.append(product_id)
            except:
                # Use a default prediction if image processing fails
                batch_ids.append(product_id)
                batch_images.append(torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE))
        else:
            # Use a default prediction if no images
            batch_ids.append(product_id)
            batch_images.append(torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE))

    if batch_images:
        batch_tensor = torch.stack(batch_images).to(device)

        with torch.no_grad():
            outputs = model(batch_tensor)
            _, predicted = torch.max(outputs, 1)

            for pid, pred_label in zip(batch_ids, predicted.cpu().numpy()):
                test_predictions.append(
                    {"_id": pid, "category_id": label_to_category[pred_label]}
                )

# Create submission DataFrame
submission_df = pd.DataFrame(test_predictions)

# Ensure we have predictions for all test samples from sample submission
sample_sub = pd.read_csv(SAMPLE_SUB)
if len(submission_df) < len(sample_sub):
    print(
        f"Warning: Missing predictions for {len(sample_sub) - len(submission_df)} samples"
    )
    # Use most frequent category for missing predictions
    most_frequent = (
        submission_df["category_id"].mode()[0]
        if not submission_df.empty
        else category_ids[0]
    )
    missing_ids = set(sample_sub["_id"]) - set(submission_df["_id"])
    missing_df = pd.DataFrame(
        [{"_id": pid, "category_id": most_frequent} for pid in missing_ids]
    )
    submission_df = pd.concat([submission_df, missing_df], ignore_index=True)

# Sort by _id to match sample submission order
submission_df = submission_df.sort_values("_id")

# Save to submission file
submission_path = "./submission/submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
print(f"Submission shape: {submission_df.shape}")

# Print final validation accuracy
print(f"\nFinal Validation Accuracy: {best_val_acc:.2f}%")

# Validate submission format
if submission_df.shape[1] == sample_sub.shape[1]:
    print("Submission format is correct!")
else:
    print("Warning: Submission format may be incorrect")

# Also check column names
if list(submission_df.columns) == list(sample_sub.columns):
    print("Column names match sample submission.")
else:
    print("Column names do not match sample submission.")
