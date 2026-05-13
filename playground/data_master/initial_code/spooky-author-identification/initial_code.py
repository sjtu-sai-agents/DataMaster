import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

# Set tokenizer parallelism to avoid warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Base seed for data splits
BASE_SEED = 42

# Load data
train_df = pd.read_csv("input/train.csv")
test_df = pd.read_csv("input/test.csv")

# Encode labels
le = LabelEncoder()
train_df["author_code"] = le.fit_transform(train_df["author"])

# Split into hold-out (10%) and training data (90%)
train_data, holdout_data = train_test_split(
    train_df,
    test_size=0.1,
    stratify=train_df["author_code"],
    random_state=BASE_SEED
)

# Split training data into train/validation (80%/20% of the 90%)
train_sub, val_sub = train_test_split(
    train_data,
    test_size=0.2,
    stratify=train_data["author_code"],
    random_state=BASE_SEED
)

# Extract texts and labels
X_train = train_sub["text"].values
y_train = train_sub["author_code"].values
X_val = val_sub["text"].values
y_val = val_sub["author_code"].values
X_holdout = holdout_data["text"].values
y_holdout = holdout_data["author_code"].values
X_test = test_df["text"].values
test_ids = test_df["id"].values

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")

# Tokenize all sets with fixed length 192
def tokenize(texts):
    return tokenizer(
        list(texts),
        max_length=192,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

train_enc = tokenize(X_train)
val_enc = tokenize(X_val)
holdout_enc = tokenize(X_holdout)
test_enc = tokenize(X_test)

# Define Dataset
class SpookyDataset(Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        item = {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

train_dataset = SpookyDataset(train_enc, y_train)
val_dataset = SpookyDataset(val_enc, y_val)
holdout_dataset = SpookyDataset(holdout_enc, y_holdout)
test_dataset = SpookyDataset(test_enc)

# Custom Trainer with R-Drop
class RDropTrainer(Trainer):
    def __init__(self, alpha=4.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    # Fix: add **kwargs to accept extra arguments like num_items_in_batch
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        if model.training:
            # Two forward passes with different dropout masks
            outputs1 = model(**inputs)
            outputs2 = model(**inputs)
            logits1 = outputs1.logits
            logits2 = outputs2.logits

            loss_fct = nn.CrossEntropyLoss()
            loss1 = loss_fct(logits1.view(-1, model.config.num_labels), labels.view(-1))
            loss2 = loss_fct(logits2.view(-1, model.config.num_labels), labels.view(-1))
            ce_loss = (loss1 + loss2) * 0.5

            # Symmetric KL divergence
            log_prob1 = F.log_softmax(logits1, dim=-1)
            prob2 = F.softmax(logits2, dim=-1)
            kl_loss1 = F.kl_div(log_prob1, prob2, reduction='batchmean')
            log_prob2 = F.log_softmax(logits2, dim=-1)
            prob1 = F.softmax(logits1, dim=-1)
            kl_loss2 = F.kl_div(log_prob2, prob1, reduction='batchmean')
            kl_loss = (kl_loss1 + kl_loss2) / 2.0

            loss = ce_loss + self.alpha * kl_loss

            if return_outputs:
                return (loss, outputs1)
            else:
                return loss
        else:
            # Evaluation mode: standard cross entropy
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            if return_outputs:
                return (loss, outputs)
            else:
                return loss

# Function to set all seeds
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# Ensemble seeds
seeds = [42, 43, 44]

# Lists to store predictions
holdout_preds = []
test_preds = []

for seed in seeds:
    print(f"\n===== Training model with seed {seed} =====")
    set_seed(seed)

    # Create output directories
    output_dir = f"./working/model_seed{seed}"
    logging_dir = f"./working/logs_seed{seed}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(logging_dir, exist_ok=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        weight_decay=0.01,
        warmup_ratio=0.1,
        fp16=True,
        logging_dir=logging_dir,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=seed,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to="none",
    )

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        "microsoft/deberta-v3-large",
        num_labels=3
    )

    # Initialize trainer
    trainer = RDropTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )

    # Train
    trainer.train()

    # Predict on holdout and test sets
    holdout_pred = trainer.predict(holdout_dataset)
    test_pred = trainer.predict(test_dataset)

    holdout_logits = holdout_pred.predictions
    test_logits = test_pred.predictions

    # Convert to probabilities
    holdout_probs = torch.softmax(torch.tensor(holdout_logits), dim=-1).numpy()
    test_probs = torch.softmax(torch.tensor(test_logits), dim=-1).numpy()

    holdout_preds.append(holdout_probs)
    test_preds.append(test_probs)

    # Cleanup
    del model, trainer
    torch.cuda.empty_cache()

# Average predictions
holdout_ensemble = np.mean(holdout_preds, axis=0)
test_ensemble = np.mean(test_preds, axis=0)

# Compute log loss on holdout set
val_log_loss = log_loss(y_holdout, holdout_ensemble)
print(f"\nHold-out validation log loss: {val_log_loss:.6f}")

# Create submission directory
os.makedirs("submission", exist_ok=True)

# Save submission file
submission_df = pd.DataFrame({
    "id": test_ids,
    "EAP": test_ensemble[:, 0],
    "HPL": test_ensemble[:, 1],
    "MWS": test_ensemble[:, 2]
})
submission_df.to_csv("submission/submission.csv", index=False)
print("Submission saved to submission/submission.csv")