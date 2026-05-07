import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")
import os
from tqdm import tqdm
import gc

# Set sharing strategy to avoid file descriptor issues
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define identity columns that are in the evaluation
IDENTITY_COLUMNS = [
    "male",
    "female",
    "homosexual_gay_or_lesbian",
    "christian",
    "jewish",
    "muslim",
    "black",
    "white",
    "psychiatric_or_mental_illness",
]

# Load data
print("Loading data...")
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

# Prepare target and identity columns
train_df["target"] = (train_df["target"] >= 0.5).astype(float)
for col in IDENTITY_COLUMNS:
    train_df[col] = train_df[col].fillna(0).apply(lambda x: 1 if x > 0.5 else 0)

# Use smaller validation split for efficiency with large dataset
train_data, val_data = train_test_split(
    train_df, test_size=0.1, random_state=SEED, stratify=train_df["target"]
)

print(f"Train size: {len(train_data)}, Val size: {len(val_data)}")


# Custom dataset class
class ToxicityDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=128):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = str(self.df.iloc[idx]["comment_text"])
        target = self.df.iloc[idx]["target"]

        # Get identity labels
        identity_labels = []
        for col in IDENTITY_COLUMNS:
            identity_labels.append(self.df.iloc[idx][col])

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "target": torch.tensor(target, dtype=torch.float),
            "identity_labels": torch.tensor(identity_labels, dtype=torch.float),
        }


# Model with bias mitigation - updated for DistilBERT
class BiasAwareBERT(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", num_identities=9):
        super(BiasAwareBERT, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)

        # Main toxicity classifier
        self.toxicity_classifier = nn.Linear(self.bert.config.hidden_size, 1)

        # Auxiliary identity classifier for bias awareness
        self.identity_classifier = nn.Linear(
            self.bert.config.hidden_size, num_identities
        )

        # Initialize weights
        nn.init.normal_(self.toxicity_classifier.weight, std=0.02)
        nn.init.normal_(self.identity_classifier.weight, std=0.02)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Use mean pooling instead of pooler_output for DistilBERT
        last_hidden_state = outputs.last_hidden_state
        # Apply attention mask to exclude padding tokens
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        pooled_output = sum_embeddings / sum_mask

        pooled_output = self.dropout(pooled_output)

        # Toxicity prediction
        toxicity_logits = self.toxicity_classifier(pooled_output)

        # Identity predictions (for bias awareness)
        identity_logits = self.identity_classifier(pooled_output)

        return toxicity_logits, identity_logits


# Initialize tokenizer and model - using DistilBERT for faster training
print("Initializing model...")
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = BiasAwareBERT(model_name, num_identities=len(IDENTITY_COLUMNS))
model = model.to(device)

# Create datasets and dataloaders with reduced workers
train_dataset = ToxicityDataset(train_data, tokenizer)
val_dataset = ToxicityDataset(val_data, tokenizer)

# Reduced num_workers to 2 to avoid file descriptor issues
train_loader = DataLoader(
    train_dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=128, shuffle=False, num_workers=2, pin_memory=True
)

# Training setup
num_epochs = 2  # Reduced for time efficiency
total_steps = len(train_loader) * num_epochs

# Use torch.optim.AdamW instead of transformers.AdamW
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, eps=1e-8)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=total_steps
)

# Loss functions
toxicity_loss_fn = nn.BCEWithLogitsLoss()
identity_loss_fn = nn.BCEWithLogitsLoss()

# Training loop
print("Starting training...")
best_val_auc = 0

for epoch in range(num_epochs):
    model.train()
    train_loss = 0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for batch in progress_bar:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True).unsqueeze(1)
        identity_targets = batch["identity_labels"].to(device, non_blocking=True)

        toxicity_logits, identity_logits = model(input_ids, attention_mask)

        # Calculate losses
        loss_toxicity = toxicity_loss_fn(toxicity_logits, targets)
        loss_identity = identity_loss_fn(identity_logits, identity_targets)

        # Combined loss with weight on identity loss for bias mitigation
        loss = loss_toxicity + 0.3 * loss_identity

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        train_loss += loss.item()
        progress_bar.set_postfix({"loss": loss.item()})

    # Validation with explicit cleanup of data loader
    model.eval()
    val_preds = []
    val_targets = []
    val_identities = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            targets = batch["target"].numpy()
            identity_labels = batch["identity_labels"].numpy()

            toxicity_logits, _ = model(input_ids, attention_mask)
            preds = torch.sigmoid(toxicity_logits).cpu().numpy()

            val_preds.extend(preds.flatten())
            val_targets.extend(targets)
            val_identities.extend(identity_labels)

    # Calculate overall AUC
    val_auc = roc_auc_score(val_targets, val_preds)

    # Calculate subgroup AUCs for important identities
    val_identities = np.array(val_identities)
    val_preds = np.array(val_preds)
    val_targets = np.array(val_targets)

    subgroup_aucs = []
    for i, col in enumerate(IDENTITY_COLUMNS):
        subgroup_mask = val_identities[:, i] == 1
        if subgroup_mask.sum() > 0 and len(np.unique(val_targets[subgroup_mask])) > 1:
            subgroup_auc = roc_auc_score(
                val_targets[subgroup_mask], val_preds[subgroup_mask]
            )
            subgroup_aucs.append(subgroup_auc)

    mean_subgroup_auc = np.mean(subgroup_aucs) if subgroup_aucs else 0

    print(f"\nEpoch {epoch+1}:")
    print(f"  Overall AUC: {val_auc:.4f}")
    print(f"  Mean Subgroup AUC: {mean_subgroup_auc:.4f}")
    print(f"  Combined Metric: {(val_auc + mean_subgroup_auc) / 2:.4f}")

    # Save best model
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        torch.save(model.state_dict(), "./working/best_model.pt")

# Clean up dataloaders to free file descriptors
del train_loader, val_loader
gc.collect()

# Load best model for final predictions
model.load_state_dict(torch.load("./working/best_model.pt"))
model.eval()

# Make predictions on test set
print("\nMaking predictions on test set...")
test_predictions = []
test_ids = test_df["id"].tolist()

# Process test data in batches
test_batch_size = 128
for i in tqdm(range(0, len(test_df), test_batch_size)):
    batch_texts = test_df["comment_text"].iloc[i : i + test_batch_size].tolist()

    encoding = tokenizer(
        batch_texts,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt",
    )

    with torch.no_grad():
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        toxicity_logits, _ = model(input_ids, attention_mask)
        batch_preds = torch.sigmoid(toxicity_logits).cpu().numpy().flatten()

        test_predictions.extend(batch_preds)

# Create submission file
submission_df = pd.DataFrame({"id": test_ids, "prediction": test_predictions})

# Ensure submission directory exists
os.makedirs("./submission", exist_ok=True)

# Save submission
submission_path = "./submission/submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"\nSubmission saved to {submission_path}")
print(f"Submission shape: {submission_df.shape}")

# Calculate final validation metric for reporting using a fresh dataloader
val_dataset_final = ToxicityDataset(val_data, tokenizer)
final_val_loader = DataLoader(
    val_dataset_final, batch_size=128, shuffle=False, num_workers=2
)

model.eval()
final_val_preds = []
final_val_targets = []

with torch.no_grad():
    for batch in final_val_loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        targets = batch["target"].numpy()

        toxicity_logits, _ = model(input_ids, attention_mask)
        preds = torch.sigmoid(toxicity_logits).cpu().numpy()

        final_val_preds.extend(preds.flatten())
        final_val_targets.extend(targets)

final_auc = roc_auc_score(final_val_targets, final_val_preds)
print(f"\nFinal Validation AUC: {final_auc:.4f}")

# Clean up
del model, final_val_loader
torch.cuda.empty_cache()
gc.collect()
