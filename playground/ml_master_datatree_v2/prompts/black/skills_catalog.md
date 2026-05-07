# Data Cleaning & Augmentation Skills Catalog

每个技能包含：**适用场景**、**依赖库**、**可直接使用的代码片段**。
选择后将代码片段复制到 `MyDataLoader.setup()` 或辅助函数中。

---

## 🖼️ 图像数据技能（Image Skills）

---

### SKILL-IMG-001: 自适应图像尺寸统一（Adaptive Resize）

**适用场景**：合并外部数据时图片尺寸不一致，DataLoader batch 时报 `stack expects each tensor to be equal size`

**依赖**：`PIL`, `torchvision`

```python
from torchvision import transforms
from PIL import Image

def build_safe_transform(img_size=224, is_train=True):
    """
    Returns a transform that always resizes to img_size first,
    preventing size mismatch errors when mixing datasets.
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]
    if is_train:
        return transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),   # resize first
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),              # always exact size
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ])

# 使用方式：替换原来的 train_transform / eval_transform
train_transform = build_safe_transform(img_size=224, is_train=True)
eval_transform  = build_safe_transform(img_size=224, is_train=False)
```

---

### SKILL-IMG-002: 损坏图像过滤（Corrupt Image Filter）

**适用场景**：合并外部数据后出现 `PIL.UnidentifiedImageError` 或随机 batch 失败

**依赖**：`PIL`

```python
import os
from PIL import Image

def filter_valid_images(image_paths: list) -> list:
    """Return only paths that can be opened as valid images."""
    valid = []
    for p in image_paths:
        try:
            with Image.open(p) as img:
                img.verify()   # verify without fully decoding
            valid.append(p)
        except Exception:
            pass
    return valid

def filter_valid_parquet_rows(df, image_col="image", bytes_key="bytes"):
    """Filter DataFrame rows where image bytes can be decoded."""
    import io
    valid_mask = []
    for _, row in df.iterrows():
        try:
            raw = row[image_col]
            if isinstance(raw, dict):
                raw = raw[bytes_key]
            Image.open(io.BytesIO(raw)).verify()
            valid_mask.append(True)
        except Exception:
            valid_mask.append(False)
    return df[valid_mask].reset_index(drop=True)
```

---

### SKILL-IMG-003: MixUp 数据增强

**适用场景**：分类任务，提升模型泛化能力，特别是多标签任务

**依赖**：`torch`, `numpy`

```python
import torch
import numpy as np

def mixup_data(x, y, alpha=0.4):
    """
    Returns mixed inputs, mixed labels, and lambda.
    x: (B, C, H, W) tensor
    y: (B, num_classes) float tensor (one-hot or soft label)
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    mixed_y = lam * y + (1 - lam) * y[index]
    return mixed_x, mixed_y

# 使用方式（在训练循环中）：
# for images, labels, _ in train_loader:
#     images, labels = images.to(device), labels.to(device)
#     images, labels = mixup_data(images, labels, alpha=0.4)
#     outputs = model(images)
#     loss = criterion(outputs, labels)
```

---

### SKILL-IMG-004: CutMix 数据增强

**适用场景**：图像分类，比 MixUp 在细粒度识别上效果更好

**依赖**：`torch`, `numpy`

```python
import torch
import numpy as np

def rand_bbox(size, lam):
    W, H = size[2], size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    x1 = max(cx - cut_w // 2, 0)
    y1 = max(cy - cut_h // 2, 0)
    x2 = min(cx + cut_w // 2, W)
    y2 = min(cy + cut_h // 2, H)
    return x1, y1, x2, y2

def cutmix_data(x, y, alpha=1.0):
    """
    x: (B, C, H, W), y: (B, num_classes) float tensor
    """
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    x1, y1, x2, y2 = rand_bbox(x.size(), lam)
    x_new = x.clone()
    x_new[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    lam_actual = 1 - (x2 - x1) * (y2 - y1) / (x.size(2) * x.size(3))
    mixed_y = lam_actual * y + (1 - lam_actual) * y[index]
    return x_new, mixed_y
```

---

### SKILL-IMG-005: CLAHE 对比度增强

**适用场景**：叶片/医疗图像，光照不均导致特征不明显

**依赖**：`opencv-python`, `PIL`, `numpy`

```python
import cv2
import numpy as np
from PIL import Image

def apply_clahe(pil_img: Image.Image, clip_limit=2.0, tile_size=(8, 8)) -> Image.Image:
    """
    Apply Contrast Limited Adaptive Histogram Equalization.
    Works on the L channel of LAB color space.
    """
    img_array = np.array(pil_img.convert("RGB"))
    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return Image.fromarray(enhanced)

# 使用方式：在 Dataset.__getitem__ 中对 PIL image 调用
# image = apply_clahe(image)   # before transform
```

---

### SKILL-IMG-006: 测试时增强 TTA（Test Time Augmentation）

**适用场景**：推理阶段，通过多次 augment 平均预测降低方差（通常 +1-3% F1）

**依赖**：`torch`, `torchvision`

```python
import torch
from torchvision import transforms

def predict_with_tta(model, loader, device, num_augments=5):
    """
    Run TTA: original + hflip + vflip + rot90 + rot180.
    Returns (all_probs_np, all_targets_np or None).
    """
    aug_fns = [
        lambda x: x,
        lambda x: torch.flip(x, dims=[3]),   # hflip
        lambda x: torch.flip(x, dims=[2]),   # vflip
        lambda x: torch.rot90(x, k=1, dims=[2, 3]),
        lambda x: torch.rot90(x, k=2, dims=[2, 3]),
    ][:num_augments]

    model.eval()
    all_probs, all_targets = [], []
    with torch.no_grad():
        for batch in loader:
            has_labels = len(batch) == 3
            if has_labels:
                images, labels, _ = batch
                all_targets.append(labels.cpu())
            else:
                images, _ = batch
            images = images.to(device)
            avg_probs = sum(
                torch.sigmoid(model(aug(images))).cpu() for aug in aug_fns
            ) / len(aug_fns)
            all_probs.append(avg_probs)

    probs = torch.cat(all_probs).numpy()
    targets = torch.cat(all_targets).numpy() if all_targets else None
    return (probs, targets) if targets is not None else probs
```

---

### SKILL-IMG-007: 类别权重采样（Weighted Sampler）

**适用场景**：类别严重不平衡，少数类预测效果差

**依赖**：`torch`

```python
import torch
from torch.utils.data import WeightedRandomSampler

def build_weighted_sampler(labels_array):
    """
    labels_array: numpy array of shape (N, num_classes) — binary multi-hot
    Returns a WeightedRandomSampler for DataLoader.
    """
    import numpy as np
    # Use the dominant class (argmax) to compute sample weights
    dominant = labels_array.argmax(axis=1)
    class_counts = np.bincount(dominant, minlength=labels_array.shape[1])
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = class_weights[dominant]
    sample_weights_tensor = torch.tensor(sample_weights, dtype=torch.double)
    return WeightedRandomSampler(
        weights=sample_weights_tensor,
        num_samples=len(sample_weights_tensor),
        replacement=True,
    )

# 使用方式：
# sampler = build_weighted_sampler(train_labels)
# train_loader = DataLoader(train_dataset, batch_size=64, sampler=sampler, ...)
```

---

## 📊 表格数据技能（Tabular Skills）

---

### SKILL-TAB-001: 缺失值智能填充

**适用场景**：CSV 数据有缺失值（NaN）

**依赖**：`pandas`, `sklearn`

```python
import pandas as pd
from sklearn.impute import KNNImputer

def impute_missing(df: pd.DataFrame, strategy="median", knn_neighbors=5) -> pd.DataFrame:
    """
    strategy: 'median' | 'mean' | 'knn'
    """
    num_cols = df.select_dtypes(include="number").columns
    if strategy == "knn":
        imputer = KNNImputer(n_neighbors=knn_neighbors)
        df[num_cols] = imputer.fit_transform(df[num_cols])
    else:
        fill = df[num_cols].median() if strategy == "median" else df[num_cols].mean()
        df[num_cols] = df[num_cols].fillna(fill)
    cat_cols = df.select_dtypes(include="object").columns
    df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])
    return df
```

---

### SKILL-TAB-002: 异常值处理（IQR Clipping）

**适用场景**：数值特征有极端异常值影响模型

**依赖**：`pandas`, `numpy`

```python
import numpy as np

def clip_outliers_iqr(df, cols=None, factor=1.5):
    """Clip values outside [Q1 - factor*IQR, Q3 + factor*IQR]."""
    cols = cols or df.select_dtypes(include="number").columns.tolist()
    for col in cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        df[col] = df[col].clip(lower=lower, upper=upper)
    return df
```

---

### SKILL-TAB-003: 标签编码与特征交叉

**适用场景**：类别特征较多，或两个特征组合有意义

**依赖**：`pandas`, `sklearn`

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def encode_categoricals(df, cols=None):
    """Label encode specified (or all) object columns."""
    cols = cols or df.select_dtypes(include="object").columns.tolist()
    for col in cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    return df

def add_cross_features(df, col_pairs: list[tuple]) -> pd.DataFrame:
    """Create interaction features from column pairs."""
    for a, b in col_pairs:
        if a in df.columns and b in df.columns:
            df[f"{a}_x_{b}"] = df[a].astype(str) + "_" + df[b].astype(str)
    return df
```

---

## 🏋️ 训练增强技能（Training Skills）

---

### SKILL-TRAIN-001: Focal Loss（处理类别不平衡）

**适用场景**：多标签分类，少数类 F1 很低

**依赖**：`torch`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Binary focal loss for multi-label classification.
    gamma=2 focuses on hard examples; alpha balances pos/neg.
    """
    def __init__(self, gamma=2.0, alpha=0.25, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        loss = focal_weight * bce
        return loss.mean() if self.reduction == "mean" else loss.sum()

# 使用方式：
# criterion = FocalLoss(gamma=2.0, alpha=0.25)
```

---

### SKILL-TRAIN-002: Label Smoothing Loss

**适用场景**：标签噪声较多，模型过于自信

**依赖**：`torch`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothingBCE(nn.Module):
    """BCE with label smoothing for multi-label classification."""
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits, targets):
        # Smooth: 0 → smoothing/2, 1 → 1 - smoothing/2
        smooth_targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        return F.binary_cross_entropy_with_logits(logits, smooth_targets)

# 使用方式：
# criterion = LabelSmoothingBCE(smoothing=0.05)
```

---

### SKILL-TRAIN-003: Cosine Annealing with Warm Restart

**适用场景**：提升收敛稳定性，避免 lr 衰减过快

**依赖**：`torch`

```python
import torch.optim as optim

def build_cosine_scheduler(optimizer, epochs, warmup_epochs=1):
    """
    Cosine LR schedule with optional linear warmup.
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs   # linear warmup
        progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
        return 0.5 * (1 + __import__('math').cos(__import__('math').pi * progress))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# 使用方式：
# scheduler = build_cosine_scheduler(optimizer, epochs=EPOCHS, warmup_epochs=1)
# # 在每个 epoch 结束后调用 scheduler.step()
```

---

### SKILL-TRAIN-004: 动态阈值优化（Per-class Threshold Tuning）

**适用场景**：多标签分类，固定 0.5 阈值导致 F1 不佳

**依赖**：`numpy`, `sklearn`

```python
import numpy as np
from sklearn.metrics import f1_score

def optimize_thresholds(probs, targets, search_range=np.arange(0.05, 0.95, 0.05)):
    """
    Find the best per-class threshold on validation set.
    probs, targets: (N, num_classes) numpy arrays.
    Returns thresholds array of shape (num_classes,).
    """
    num_classes = probs.shape[1]
    best_thresholds = np.full(num_classes, 0.5)
    for i in range(num_classes):
        best_f1, best_t = 0.0, 0.5
        for t in search_range:
            preds = (probs[:, i] >= t).astype(int)
            f1 = f1_score(targets[:, i], preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        best_thresholds[i] = best_t
    return best_thresholds

def apply_thresholds(probs, thresholds):
    """Apply per-class thresholds, ensure at least one label per sample."""
    preds = np.zeros_like(probs, dtype=int)
    for i, t in enumerate(thresholds):
        preds[:, i] = (probs[:, i] >= t).astype(int)
    # Fallback: if no label predicted, use argmax
    empty = preds.sum(axis=1) == 0
    if empty.any():
        preds[empty, probs[empty].argmax(axis=1)] = 1
    return preds
```
