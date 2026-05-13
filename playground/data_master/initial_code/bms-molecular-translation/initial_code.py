import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from tqdm import tqdm
import warnings
from collections import Counter
import Levenshtein

warnings.filterwarnings("ignore")

# Set device and random seeds
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)

# Paths
INPUT_DIR = "./input"
TRAIN_IMAGE_DIR = os.path.join(INPUT_DIR, "train")
TEST_IMAGE_DIR = os.path.join(INPUT_DIR, "test")
TRAIN_LABELS_PATH = os.path.join(INPUT_DIR, "train_labels.csv")
SAMPLE_SUB_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")
SUBMISSION_DIR = "./submission"
WORKING_DIR = "./working"
os.makedirs(SUBMISSION_DIR, exist_ok=True)
os.makedirs(WORKING_DIR, exist_ok=True)
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Hyperparameters
BATCH_SIZE = 64
EPOCHS = 3
LEARNING_RATE = 3e-4
IMG_SIZE = 224
MAX_LEN = 200
EMBED_DIM = 256
HIDDEN_DIM = 512
NUM_WORKERS = 8

# Load and prepare data
print("Loading data...")
train_df = pd.read_csv(TRAIN_LABELS_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

# Use subset for faster training given time constraints
train_df = train_df.sample(50000, random_state=42).reset_index(drop=True)

# Create vocabulary from training InChI strings
all_text = " ".join(train_df["InChI"].astype(str))
char_counts = Counter(all_text)
vocab = sorted(char_counts.keys())

# Special tokens
PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"

vocab = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN] + vocab
char2idx = {c: i for i, c in enumerate(vocab)}
idx2char = {i: c for i, c in enumerate(vocab)}
VOCAB_SIZE = len(vocab)
PAD_IDX = char2idx[PAD_TOKEN]
SOS_IDX = char2idx[SOS_TOKEN]
EOS_IDX = char2idx[EOS_TOKEN]
UNK_IDX = char2idx[UNK_TOKEN]

print(f"Vocabulary size: {VOCAB_SIZE}")


# Dataset class
class MolecularDataset(Dataset):
    def __init__(self, df, image_dir, transform=None, is_test=False):
        self.df = df
        self.image_dir = image_dir
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row["image_id"]

        # Build image path from 3-level folder structure
        img_path = os.path.join(
            self.image_dir, image_id[0], image_id[1], image_id[2], f"{image_id}.png"
        )

        try:
            image = Image.open(img_path).convert("RGB")
        except:
            # Create blank image if file not found
            image = Image.new("RGB", (IMG_SIZE, IMG_SIZE), color=(255, 255, 255))

        if self.transform:
            image = self.transform(image)

        if self.is_test:
            return image, image_id

        # Encode InChI string
        inchi = row["InChI"]
        encoded = [SOS_IDX] + [char2idx.get(c, UNK_IDX) for c in inchi] + [EOS_IDX]
        encoded = encoded[:MAX_LEN]
        if len(encoded) < MAX_LEN:
            encoded += [PAD_IDX] * (MAX_LEN - len(encoded))

        return image, torch.tensor(encoded, dtype=torch.long)


# Transformations
train_transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

val_transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Split data
train_size = int(0.8 * len(train_df))
val_size = len(train_df) - train_size
train_data = train_df[:train_size].reset_index(drop=True)
val_data = train_df[train_size:].reset_index(drop=True)

train_dataset = MolecularDataset(train_data, TRAIN_IMAGE_DIR, train_transform)
val_dataset = MolecularDataset(val_data, TRAIN_IMAGE_DIR, val_transform)

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

print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")


# Model architecture
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b0", pretrained=True, num_classes=0, global_pool=""
        )
        self.projection = nn.Linear(1280, HIDDEN_DIM)

    def forward(self, x):
        features = self.backbone(x)
        features = features.mean(dim=[2, 3])  # Global average pooling
        return self.projection(features)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        self.lstm = nn.LSTM(
            EMBED_DIM + HIDDEN_DIM,
            HIDDEN_DIM,
            batch_first=True,
            num_layers=2,
            dropout=0.3,
        )
        self.fc = nn.Linear(HIDDEN_DIM, VOCAB_SIZE)

    def forward(
        self, encoder_output, target_seq=None, max_len=MAX_LEN, teacher_forcing=False
    ):
        batch_size = encoder_output.size(0)

        if teacher_forcing and target_seq is not None:
            # Teacher forcing: use ground truth as input
            seq_len = target_seq.size(1) - 1
            embedded = self.embedding(target_seq[:, :-1])
            encoder_output = encoder_output.unsqueeze(1).repeat(1, seq_len, 1)
            lstm_input = torch.cat([embedded, encoder_output], dim=-1)
            output, _ = self.lstm(lstm_input)
            return self.fc(output)  # Shape: (batch_size, seq_len, VOCAB_SIZE)
        else:
            # Inference mode
            hidden = None
            input_token = torch.full(
                (batch_size, 1), SOS_IDX, device=device, dtype=torch.long
            )
            outputs = []

            for _ in range(max_len):
                embedded = self.embedding(input_token)
                encoder_expanded = encoder_output.unsqueeze(1)
                lstm_input = torch.cat([embedded, encoder_expanded], dim=-1)
                output, hidden = self.lstm(lstm_input, hidden)
                logits = self.fc(output.squeeze(1))
                next_token = logits.argmax(-1).unsqueeze(1)
                outputs.append(next_token)
                input_token = next_token

                # Stop if all sequences generated EOS
                if (next_token == EOS_IDX).all():
                    break

            return (
                torch.cat(outputs, dim=1)
                if outputs
                else torch.empty(batch_size, 0, device=device)
            )


class MolecularModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, images, targets=None, teacher_forcing=False):
        encoder_out = self.encoder(images)
        decoder_out = self.decoder(
            encoder_out, targets, teacher_forcing=teacher_forcing
        )
        return decoder_out


# Initialize model
model = MolecularModel().to(device)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# Training loop
print("Starting training...")
best_val_loss = float("inf")

for epoch in range(EPOCHS):
    # Training
    model.train()
    train_loss = 0
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")

    for images, targets in train_bar:
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad()
        output = model(images, targets, teacher_forcing=True)

        # Reshape for loss: output is (batch_size, seq_len, VOCAB_SIZE), targets[:, 1:] is (batch_size, seq_len)
        loss = criterion(output.reshape(-1, VOCAB_SIZE), targets[:, 1:].reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        train_loss += loss.item()
        train_bar.set_postfix({"loss": loss.item()})

    avg_train_loss = train_loss / len(train_loader)

    # Validation
    model.eval()
    val_loss = 0
    val_predictions = []
    val_targets = []

    with torch.no_grad():
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]")
        for images, targets in val_bar:
            images, targets = images.to(device), targets.to(device)

            # Compute loss with teacher forcing
            output = model(images, targets, teacher_forcing=True)
            loss = criterion(output.reshape(-1, VOCAB_SIZE), targets[:, 1:].reshape(-1))
            val_loss += loss.item()

            # Generate predictions for validation metric (without teacher forcing)
            encoder_out = model.encoder(images)
            pred_tokens = model.decoder(encoder_out, max_len=MAX_LEN)

            for i in range(len(targets)):
                # Convert predicted tokens to string
                pred_seq = pred_tokens[i].cpu().numpy()
                pred_chars = []
                for token in pred_seq:
                    if token == EOS_IDX:
                        break
                    if token in idx2char and token not in [PAD_IDX, SOS_IDX]:
                        pred_chars.append(idx2char[token])
                pred_inchi = "".join(pred_chars)
                val_predictions.append(pred_inchi)

                # Convert target tokens to string
                target_seq = targets[i].cpu().numpy()
                target_chars = []
                for token in target_seq:
                    if token == EOS_IDX:
                        break
                    if token in idx2char and token not in [PAD_IDX, SOS_IDX]:
                        target_chars.append(idx2char[token])
                target_inchi = "".join(target_chars)
                val_targets.append(target_inchi)

    avg_val_loss = val_loss / len(val_loader)

    # Calculate Levenshtein distance (competition metric)
    lev_distances = []
    for pred, target in zip(val_predictions, val_targets):
        if len(pred) == 0 or len(target) == 0:
            lev_distances.append(max(len(pred), len(target)))
        else:
            lev_distances.append(Levenshtein.distance(pred, target))

    avg_levenshtein = np.mean(lev_distances)

    print(
        f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
        f"Val Levenshtein: {avg_levenshtein:.4f}"
    )

    scheduler.step()

    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), os.path.join(WORKING_DIR, "best_model.pth"))

# Load best model for inference
model.load_state_dict(torch.load(os.path.join(WORKING_DIR, "best_model.pth")))
model.eval()

# Create test dataset
test_df = sample_sub.copy()
test_dataset = MolecularDataset(test_df, TEST_IMAGE_DIR, val_transform, is_test=True)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

# Generate predictions
print("Generating test predictions...")
predictions = []
image_ids = []

with torch.no_grad():
    test_bar = tqdm(test_loader, desc="Inference")
    for images, ids in test_bar:
        images = images.to(device)
        encoder_out = model.encoder(images)
        output_tokens = model.decoder(encoder_out, max_len=MAX_LEN)

        for i in range(len(ids)):
            tokens = output_tokens[i].cpu().numpy()
            chars = []
            for token in tokens:
                if token == EOS_IDX:
                    break
                if token in idx2char and token not in [PAD_IDX, SOS_IDX]:
                    chars.append(idx2char[token])

            pred_inchi = "".join(chars)
            if not pred_inchi.startswith("InChI="):
                pred_inchi = "InChI=1S" + pred_inchi

            predictions.append(pred_inchi)
            image_ids.append(ids[i])

# Create submission file
submission = pd.DataFrame({"image_id": image_ids, "InChI": predictions})

# Ensure all test images are included
if len(submission) < len(test_df):
    # Fill missing predictions with a default
    all_image_ids = test_df["image_id"].tolist()
    submitted_ids = set(submission["image_id"])
    missing_ids = [img_id for img_id in all_image_ids if img_id not in submitted_ids]

    for img_id in missing_ids:
        submission = pd.concat(
            [submission, pd.DataFrame({"image_id": [img_id], "InChI": ["InChI=1S"]})],
            ignore_index=True,
        )

# Sort by image_id to match sample submission order
submission = submission.sort_values("image_id").reset_index(drop=True)
submission.to_csv(SUBMISSION_PATH, index=False)
print(f"Submission saved to {SUBMISSION_PATH}")
print(f"Submission shape: {submission.shape}")
print(f"Sample predictions:\n{submission.head()}")

# Final validation metric
print(f"\nFinal Validation Levenshtein Distance: {avg_levenshtein:.4f}")
print("Note: Lower Levenshtein distance is better (0 means perfect match).")
