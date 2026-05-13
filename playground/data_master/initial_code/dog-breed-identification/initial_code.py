import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from tqdm import tqdm

# 1. 基础配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 3 # 演示用，建议实际训练 10+

# 2. 数据处理
labels_df = pd.read_csv('./input/labels.csv')
sample_sub = pd.read_csv('./input/sample_submission.csv')

# 自动获取类别并排序（确保与 submission 列顺序一致）
breeds = sample_sub.columns[1:].tolist()
breed_to_idx = {breed: i for i, breed in enumerate(breeds)}
labels_df['label'] = labels_df['breed'].map(breed_to_idx)

# 3. 极简 Dataset
class DogDataset(Dataset):
    def __init__(self, ids, labels=None, img_dir='./input/train/', transform=None):
        self.ids = ids
        self.labels = labels
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_path = os.path.join(self.img_dir, f"{img_id}.jpg")
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        if self.labels is not None:
            return image, self.labels[idx]
        return image, img_id

# 4. 转换与加载
data_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_loader = DataLoader(
    DogDataset(labels_df['id'].values, labels_df['label'].values, transform=data_transform),
    batch_size=BATCH_SIZE, shuffle=True
)

# 5. 模型与优化器
model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=len(breeds)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# 6. 极简训练
print("Training...")
for epoch in range(EPOCHS):
    model.train()
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()

# 7. 生成预测 (Submission)
print("\nPredicting...")
model.eval()
test_ids = sample_sub['id'].values
test_loader = DataLoader(
    DogDataset(test_ids, img_dir='./input/test/', transform=data_transform),
    batch_size=BATCH_SIZE, shuffle=False
)

all_preds = []
with torch.no_grad():
    for images, _ in tqdm(test_loader):
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1) # 关键：转为概率
        all_preds.append(probs.cpu().numpy())

# 8. 保存结果
submission = pd.DataFrame(np.vstack(all_preds), columns=breeds)
submission.insert(0, 'id', test_ids)
submission.to_csv('submission.csv', index=False)

print("\nDone! 'submission.csv' has been saved.")