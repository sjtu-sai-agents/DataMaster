import os
import random
import math
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import timm
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

# ----------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42)

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
INPUT_DIR = "./input"
TRAIN_CSV = os.path.join(INPUT_DIR, "train.csv")
TRAIN_IMG_DIR = os.path.join(INPUT_DIR, "train")
TEST_IMG_DIR = os.path.join(INPUT_DIR, "test")
SUBMISSION_DIR = "./submission"
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
WORKING_DIR = "./working"
os.makedirs(SUBMISSION_DIR, exist_ok=True)
os.makedirs(WORKING_DIR, exist_ok=True)

# ----------------------------------------------------------------------
# Load and prepare data
# ----------------------------------------------------------------------
df = pd.read_csv(TRAIN_CSV)
print("Total training samples:", len(df))

# Exclude new_whale from training
known_df = df[df['Id'] != 'new_whale'].copy()
print("Known whales:", len(known_df), "samples,", known_df['Id'].nunique(), "classes")

# Label encoding for known whales
unique_ids = sorted(known_df['Id'].unique())
label_to_idx = {id: i for i, id in enumerate(unique_ids)}
idx_to_label = {i: id for id, i in label_to_idx.items()}
known_df['label'] = known_df['Id'].map(label_to_idx)

# Fix: handle classes with only one sample to avoid stratification error
counts = known_df['Id'].value_counts()
singleton_ids = counts[counts == 1].index.tolist()
multi_df = known_df[~known_df['Id'].isin(singleton_ids)].copy()
singleton_df = known_df[known_df['Id'].isin(singleton_ids)].copy()

# Random split (stratification removed because number of classes > test size)
train_multi, val_multi = train_test_split(
    multi_df, test_size=0.2, random_state=42
)
# Combine: all singletons go to training
train_df = pd.concat([train_multi, singleton_df], ignore_index=True)
val_df = val_multi.copy()

print("Train size:", len(train_df), "Val size:", len(val_df))

# ----------------------------------------------------------------------
# Transforms
# ----------------------------------------------------------------------
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop((224, 224)),
    transforms.RandAugment(num_ops=3, magnitude=12),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

tta_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.FiveCrop(224),
    transforms.Lambda(lambda crops: torch.stack([
        transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])(crop) for crop in crops
    ]))
])

# ----------------------------------------------------------------------
# Datasets and DataLoaders
# ----------------------------------------------------------------------
class WhaleDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['Image']
        label = row['label']
        img_path = os.path.join(self.img_dir, img_name)
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            dummy = torch.zeros(3, 224, 224)
            return dummy, -1

class TTADataset(Dataset):
    def __init__(self, img_paths, transform=None):
        self.img_paths = img_paths
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                images = self.transform(image)   # shape [5,3,224,224]
                return images
            else:
                return torch.zeros(5, 3, 224, 224)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return torch.zeros(5, 3, 224, 224)

train_dataset = WhaleDataset(train_df, TRAIN_IMG_DIR, transform=train_transform)
val_dataset = WhaleDataset(val_df, TRAIN_IMG_DIR, transform=val_transform)

batch_size = 24
num_workers = 8
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True,
    num_workers=num_workers, pin_memory=True, drop_last=True
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False,
    num_workers=num_workers, pin_memory=True
)

# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------
class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.5, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if not self.easy_margin:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output

class WhaleModel(nn.Module):
    def __init__(self, num_classes):
        super(WhaleModel, self).__init__()
        self.backbone = timm.create_model('efficientnet_b5', pretrained=True, num_classes=0, global_pool='avg')
        in_features = self.backbone.num_features  # 2048
        self.bn1 = nn.BatchNorm1d(in_features)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(in_features, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.arcface = ArcMarginProduct(1024, num_classes)

    def forward(self, x, label=None):
        x = self.backbone(x)
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.bn2(x)
        embedding = F.normalize(x)
        if label is not None:
            logits = self.arcface(embedding, label)
            return logits, embedding
        return embedding

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

num_classes = len(unique_ids)
model = WhaleModel(num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25)
scaler = torch.cuda.amp.GradScaler()

# ----------------------------------------------------------------------
# Training and validation
# ----------------------------------------------------------------------
def evaluate(model, loader):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            with torch.cuda.amp.autocast():
                logits, _ = model(images, labels)
                loss = criterion(logits, labels)
            total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)

epochs = 25
best_val_loss = float('inf')
checkpoint_path = os.path.join(WORKING_DIR, 'best_model.pth')

for epoch in range(1, epochs+1):
    model.train()
    train_loss = 0.0
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{epochs}')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            logits, _ = model(images, labels)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item() * images.size(0)
        pbar.set_postfix(loss=loss.item())
    train_loss /= len(train_dataset)

    val_loss = evaluate(model, val_loader)
    print(f"Epoch {epoch}: train loss = {train_loss:.4f}, val loss = {val_loss:.4f}")
    scheduler.step()

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), checkpoint_path)
        print("Saved best model.")

# Load best model
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# ----------------------------------------------------------------------
# Embedding extraction
# ----------------------------------------------------------------------
def extract_embeddings(loader, is_tta=False):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for batch in tqdm(loader):
            if is_tta:
                images = batch  # [B,5,3,224,224]
                B = images.size(0)
                images = images.view(-1, 3, 224, 224).to(device)
                with torch.cuda.amp.autocast():
                    emb = model(images)  # [B*5,1024]
                emb = emb.view(B, 5, -1).mean(dim=1)  # average over crops
            else:
                images, labels = batch
                images = images.to(device)
                with torch.cuda.amp.autocast():
                    emb = model(images)
            embeddings.append(emb.cpu().numpy())
    return np.vstack(embeddings)

# Extract for all known training images (single crop)
full_known_dataset = WhaleDataset(known_df, TRAIN_IMG_DIR, transform=val_transform)
full_known_loader = DataLoader(
    full_known_dataset, batch_size=batch_size, shuffle=False,
    num_workers=num_workers, pin_memory=True
)
train_embeddings = extract_embeddings(full_known_loader)
train_ids = known_df['Id'].values

# Extract for validation images (single crop)
val_embeddings = extract_embeddings(val_loader)
val_labels = val_df['Id'].values

# ----------------------------------------------------------------------
# Threshold optimization (simulate new_whale)
# ----------------------------------------------------------------------
all_val_classes = set(val_labels)
n_unknown = int(0.2 * len(all_val_classes))
unknown_classes = set(random.sample(list(all_val_classes), n_unknown))
known_classes_sim = all_val_classes - unknown_classes

# Filter training embeddings: remove classes that are treated as unknown
train_mask = [id not in unknown_classes for id in train_ids]
train_emb_sim = train_embeddings[train_mask]
train_ids_sim = train_ids[train_mask]

# Build NN on filtered set
nn_model = NearestNeighbors(n_neighbors=50, metric='cosine')
nn_model.fit(train_emb_sim)

# Get neighbors for validation images
distances, indices = nn_model.kneighbors(val_embeddings)
similarities = 1 - distances

# Adjust ground truth: images from unknown_classes should have true label = 'new_whale'
sim_truths = ['new_whale' if id in unknown_classes else id for id in val_labels]

# MAP@5 implementation
def map5(preds, truths):
    total = 0.0
    for pred, truth in zip(preds, truths):
        for k, p in enumerate(pred):
            if p == truth:
                total += 1.0 / (k + 1)
                break
    return total / len(truths)

def predict_top5(similarities, neighbor_ids, threshold):
    preds = []
    max_sim = similarities[0] if len(similarities) > 0 else 0
    if max_sim < threshold:
        preds.append('new_whale')
    seen = set()
    for sim, nid in zip(similarities, neighbor_ids):
        if nid not in seen:
            seen.add(nid)
            preds.append(nid)
        if len(preds) >= 5:
            break
    while len(preds) < 5:
        preds.append('new_whale')
    return preds[:5]

# Search best threshold
thresholds = np.arange(0.3, 0.81, 0.05)
best_map = 0.0
best_th = 0.5

for th in thresholds:
    val_preds = []
    for i in range(len(val_embeddings)):
        sims = similarities[i]
        nbr_ids = train_ids_sim[indices[i]]
        pred = predict_top5(sims, nbr_ids, th)
        val_preds.append(pred)
    map_val = map5(val_preds, sim_truths)
    print(f"Threshold {th:.2f} -> MAP@5 = {map_val:.4f}")
    if map_val > best_map:
        best_map = map_val
        best_th = th

print(f"\nBest threshold: {best_th:.2f}  Validation MAP@5: {best_map:.4f}\n")

# ----------------------------------------------------------------------
# Test predictions and submission
# ----------------------------------------------------------------------
# List test images
test_img_names = sorted([f for f in os.listdir(TEST_IMG_DIR) if f.endswith('.jpg')])
test_img_paths = [os.path.join(TEST_IMG_DIR, f) for f in test_img_names]
test_dataset = TTADataset(test_img_paths, transform=tta_transform)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False,
    num_workers=num_workers, pin_memory=True
)

# Extract test embeddings with TTA
test_embeddings = extract_embeddings(test_loader, is_tta=True)

# Build NN on full training set
nn_full = NearestNeighbors(n_neighbors=50, metric='cosine')
nn_full.fit(train_embeddings)
distances, indices = nn_full.kneighbors(test_embeddings)
similarities = 1 - distances

# Generate predictions with best threshold
test_preds = []
for i in range(len(test_embeddings)):
    sims = similarities[i]
    nbr_ids = train_ids[indices[i]]
    pred = predict_top5(sims, nbr_ids, best_th)
    test_preds.append(pred)

# Create submission
submission = pd.DataFrame({
    'Image': test_img_names,
    'Id': [' '.join(p) for p in test_preds]
})
submission.to_csv(SUBMISSION_PATH, index=False)
print(f"Submission saved to {SUBMISSION_PATH}")