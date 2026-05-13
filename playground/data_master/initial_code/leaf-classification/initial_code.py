import os
import random
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.utils.class_weight import compute_class_weight

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ------------------------------
# 1. Load and preprocess data
# ------------------------------
train_df = pd.read_csv('./input/train.csv')
test_df = pd.read_csv('./input/test.csv')

# Tabular features: all columns except id and species
feature_cols = [col for col in train_df.columns if col not in ['id', 'species']]
X_train_tab = train_df[feature_cols].values.astype(np.float32)
X_test_tab = test_df[feature_cols].values.astype(np.float32)

# Scale features
scaler = StandardScaler()
X_train_tab_scaled = scaler.fit_transform(X_train_tab)
X_test_tab_scaled = scaler.transform(X_test_tab)

# Encode labels
le = LabelEncoder()
y_train = le.fit_transform(train_df['species'])
num_classes = len(le.classes_)
class_names = le.classes_

# Class weights (for loss)
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

# ------------------------------
# 2. Extract image features
# ------------------------------
image_dir = './input/images'

# Load pre-trained EfficientNet-B0, remove classification head
img_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
img_model.classifier = nn.Identity()  # output: 1280-dimensional pooled features
img_model = img_model.to(device)
img_model.eval()
preprocess = models.EfficientNet_B0_Weights.DEFAULT.transforms()

def extract_image_features(ids):
    features = []
    for id_val in ids:
        img_path = os.path.join(image_dir, f"{id_val}.jpg")
        img = Image.open(img_path).convert('RGB')
        img_tensor = preprocess(img).unsqueeze(0).to(device)  # (1, 3, 224, 224)
        with torch.no_grad():
            feat = img_model(img_tensor)  # (1, 1280)
        features.append(feat.cpu().numpy().flatten())
    return np.array(features, dtype=np.float32)

train_ids = train_df['id'].values
test_ids = test_df['id'].values

print("Extracting image features for training set...")
train_img_feats = extract_image_features(train_ids)
print("Extracting image features for test set...")
test_img_feats = extract_image_features(test_ids)

# Combine tabular and image features
X_train = np.hstack([X_train_tab_scaled, train_img_feats])
X_test = np.hstack([X_test_tab_scaled, test_img_feats])

# ------------------------------
# 3. Train/validation split
# ------------------------------
X_train_full, X_val, y_train_full, y_val = train_test_split(
    X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
)

# ------------------------------
# 4. PyTorch datasets and dataloaders
# ------------------------------
class LeafDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        else:
            return self.X[idx]

train_dataset = LeafDataset(X_train_full, y_train_full)
val_dataset = LeafDataset(X_val, y_val)
test_dataset = LeafDataset(X_test)

batch_size = 32
num_workers = 4
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, pin_memory=True)

# ------------------------------
# 5. Define the classifier model
# ------------------------------
class LeafModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.drop1 = nn.Dropout(0.4)

        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.drop2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.drop3 = nn.Dropout(0.2)

        self.out = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.drop1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.drop2(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.drop3(x)

        x = self.out(x)
        return x

model = LeafModel(input_dim=X_train.shape[1], num_classes=num_classes).to(device)

# Loss, optimizer, scheduler
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)

# ------------------------------
# 6. Training with early stopping
# ------------------------------
max_epochs = 200
patience = 10
best_val_loss = float('inf')
patience_counter = 0
best_state = None

for epoch in range(max_epochs):
    model.train()
    train_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * X_batch.size(0)
    train_loss /= len(train_loader.dataset)

    model.eval()
    val_loss = 0.0
    val_preds = []
    val_labels = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item() * X_batch.size(0)
            probs = torch.softmax(outputs, dim=1)
            val_preds.append(probs.cpu().numpy())
            val_labels.append(y_batch.cpu().numpy())
    val_loss /= len(val_loader.dataset)
    val_preds = np.vstack(val_preds)
    val_labels = np.concatenate(val_labels)
    val_log = log_loss(val_labels, val_preds, labels=np.arange(num_classes))

    scheduler.step(val_loss)

    print(f"Epoch {epoch+1:3d}/{max_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val LogLoss: {val_log:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        best_state = model.state_dict().copy()
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

# Load best model
model.load_state_dict(best_state)
model.eval()

# ------------------------------
# 7. Compute final validation metric
# ------------------------------
val_preds = []
val_labels = []
with torch.no_grad():
    for X_batch, y_batch in val_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)
        probs = torch.softmax(outputs, dim=1)
        val_preds.append(probs.cpu().numpy())
        val_labels.append(y_batch.cpu().numpy())
val_preds = np.vstack(val_preds)
val_labels = np.concatenate(val_labels)
final_val_log = log_loss(val_labels, val_preds, labels=np.arange(num_classes))
print(f"\nFinal Validation Log Loss: {final_val_log:.5f}")

# ------------------------------
# 8. Generate test predictions
# ------------------------------
test_preds = []
with torch.no_grad():
    for X_batch in test_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        probs = torch.softmax(outputs, dim=1)
        test_preds.append(probs.cpu().numpy())
test_preds = np.vstack(test_preds)

# Clip and renormalize (safeguard for log)
test_preds = np.clip(test_preds, 1e-15, 1-1e-15)
test_preds = test_preds / test_preds.sum(axis=1, keepdims=True)

# Create submission dataframe
submission = pd.DataFrame(test_preds, columns=class_names)
submission.insert(0, 'id', test_df['id'].values)

# Ensure submission directory exists
os.makedirs('./submission', exist_ok=True)
submission.to_csv('./submission/submission.csv', index=False)
print("Submission saved to ./submission/submission.csv")

print("Done.")