import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    get_linear_schedule_with_warmup
)
import torch.cuda.amp as amp
from torch.optim import AdamW

# ---------- Constants ----------
SEED = 42
MODEL_NAME = "deepset/xlm-roberta-large-squad2"
MAX_LENGTH = 512
DOC_STRIDE = 128
TRAIN_BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 4
EVAL_BATCH_SIZE = 4
EPOCHS = 2
LEARNING_RATE = 1e-5
WARMUP_RATIO = 0.1
N_BEST = 20
MAX_ANSWER_LENGTH = 64

# ---------- Set Seeds ----------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(SEED)

# ---------- Device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------- Load Data ----------
train_df = pd.read_csv("input/train.csv")
test_df = pd.read_csv("input/test.csv")

# ---------- Data Cleaning ----------
def clip_answer_start(row):
    context = row["context"]
    length = len(context)
    start = int(row["answer_start"])
    if length == 0:
        return 0
    return max(0, min(start, length - 1))

# Fill NaNs and clean
for col in ["context", "question", "answer_text", "language"]:
    train_df[col] = train_df[col].fillna("")
    if col in test_df:
        test_df[col] = test_df[col].fillna("")
train_df["answer_start"] = train_df["answer_start"].fillna(0).astype(int)
train_df["answer_start"] = train_df.apply(clip_answer_start, axis=1)

# ---------- Train-Validation Split ----------
train_split, val_split = train_test_split(
    train_df,
    test_size=0.2,
    stratify=train_df["language"],
    random_state=SEED
)

# ---------- Tokenizer ----------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
pad_on_right = tokenizer.padding_side == "right"

# ---------- Feature Preparation ----------
def prepare_train_features(examples):
    features = []
    for _, row in tqdm(examples.iterrows(), total=len(examples), desc="Preparing train features"):
        context = row["context"]
        question = row["question"]
        answer_text = row["answer_text"]
        answer_start = row["answer_start"]
        answer_end = answer_start + len(answer_text)

        tokenized = tokenizer(
            question if pad_on_right else context,
            context if pad_on_right else question,
            truncation="only_second" if pad_on_right else "only_first",
            max_length=MAX_LENGTH,
            stride=DOC_STRIDE,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        sample_mapping = tokenized.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized.pop("offset_mapping")

        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized["input_ids"][i]
            attention_mask = tokenized["attention_mask"][i]
            token_type_ids = tokenized.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids[i]

            sequence_ids = tokenized.sequence_ids(i)
            cls_index = input_ids.index(tokenizer.cls_token_id)

            # Find first and last context token
            context_start = 0
            while sequence_ids[context_start] != (1 if pad_on_right else 0):
                context_start += 1
            context_end = len(input_ids) - 1
            while sequence_ids[context_end] != (1 if pad_on_right else 0):
                context_end -= 1

            # Check if answer is inside this window
            if not (offsets[context_start][0] <= answer_start and offsets[context_end][1] >= answer_end):
                start_position = cls_index
                end_position = cls_index
            else:
                # Find token indices
                token_start_index = context_start
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= answer_start:
                    token_start_index += 1
                start_position = token_start_index - 1

                token_end_index = context_end
                while offsets[token_end_index][1] >= answer_end:
                    token_end_index -= 1
                end_position = token_end_index + 1

                # Sanity check
                if start_position < context_start or end_position > context_end or start_position > end_position:
                    start_position = cls_index
                    end_position = cls_index

            feature = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "start_positions": start_position,
                "end_positions": end_position,
            }
            if token_type_ids is not None:
                feature["token_type_ids"] = token_type_ids
            features.append(feature)
    return features

def prepare_eval_features(examples):
    features = []
    for _, row in tqdm(examples.iterrows(), total=len(examples), desc="Preparing eval features"):
        context = row["context"]
        question = row["question"]
        example_id = row["id"]

        tokenized = tokenizer(
            question if pad_on_right else context,
            context if pad_on_right else question,
            truncation="only_second" if pad_on_right else "only_first",
            max_length=MAX_LENGTH,
            stride=DOC_STRIDE,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        sample_mapping = tokenized.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized.pop("offset_mapping")

        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized["input_ids"][i]
            attention_mask = tokenized["attention_mask"][i]
            token_type_ids = tokenized.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids[i]

            sequence_ids = tokenized.sequence_ids(i)
            # Keep only context offsets, set others to None
            offset = []
            for si, (start, end) in zip(sequence_ids, offsets):
                if si == (1 if pad_on_right else 0):
                    offset.append((start, end))
                else:
                    offset.append(None)

            feature = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "offset_mapping": offset,
                "example_id": example_id,
            }
            if token_type_ids is not None:
                feature["token_type_ids"] = token_type_ids
            features.append(feature)
    return features

# ---------- Dataset Classes ----------
class TrainQADataset(Dataset):
    def __init__(self, features):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        f = self.features[idx]
        item = {
            "input_ids": torch.tensor(f["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(f["attention_mask"], dtype=torch.long),
            "start_positions": torch.tensor(f["start_positions"], dtype=torch.long),
            "end_positions": torch.tensor(f["end_positions"], dtype=torch.long),
        }
        if "token_type_ids" in f:
            item["token_type_ids"] = torch.tensor(f["token_type_ids"], dtype=torch.long)
        return item

class EvalQADataset(Dataset):
    def __init__(self, features):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        f = self.features[idx]
        item = {
            "input_ids": torch.tensor(f["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(f["attention_mask"], dtype=torch.long),
            "offset_mapping": f["offset_mapping"],
            "example_id": f["example_id"],
        }
        if "token_type_ids" in f:
            item["token_type_ids"] = torch.tensor(f["token_type_ids"], dtype=torch.long)
        return item

# ---------- Custom Collate for Evaluation ----------
def eval_collate_fn(batch):
    """Collate function for evaluation: stack tensors, keep offset_mapping and example_id as lists."""
    collated = {}
    # Stack tensor fields
    collated["input_ids"] = torch.stack([item["input_ids"] for item in batch])
    collated["attention_mask"] = torch.stack([item["attention_mask"] for item in batch])
    if "token_type_ids" in batch[0]:
        collated["token_type_ids"] = torch.stack([item["token_type_ids"] for item in batch])
    # Keep offset_mapping and example_id as lists
    collated["offset_mapping"] = [item["offset_mapping"] for item in batch]
    collated["example_id"] = [item["example_id"] for item in batch]
    return collated

# ---------- Jaccard Metric ----------
def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    if len(a) == 0 and len(b) == 0:
        return 1.0
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

def compute_metric(predictions, ground_truth_df):
    scores = []
    for _, row in ground_truth_df.iterrows():
        pred = predictions.get(row["id"], "")
        true = row["answer_text"]
        scores.append(jaccard(pred, true))
    return np.mean(scores)

# ---------- Prediction Function ----------
def predict_answers(model, eval_features, eval_examples):
    model.eval()
    eval_dataset = EvalQADataset(eval_features)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
        collate_fn=eval_collate_fn  # use custom collate
    )

    all_start_logits = []
    all_end_logits = []
    all_offset_mappings = []
    all_example_ids = []

    for batch in tqdm(eval_loader, desc="Running inference"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)

        with torch.no_grad(), amp.autocast():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
        start_logits = outputs.start_logits.cpu().numpy()
        end_logits = outputs.end_logits.cpu().numpy()

        all_start_logits.append(start_logits)
        all_end_logits.append(end_logits)
        all_offset_mappings.extend(batch["offset_mapping"])
        all_example_ids.extend(batch["example_id"])

    start_logits = np.concatenate(all_start_logits, axis=0)
    end_logits = np.concatenate(all_end_logits, axis=0)

    # Map example_id to list of feature indices
    example_to_features = {}
    for idx, ex_id in enumerate(all_example_ids):
        example_to_features.setdefault(ex_id, []).append(idx)

    id_to_context = {row["id"]: row["context"] for _, row in eval_examples.iterrows()}
    predictions = {}

    for ex_id, feat_indices in tqdm(example_to_features.items(), desc="Post-processing"):
        context = id_to_context[ex_id]
        best_answer = ""
        best_score = -float('inf')

        for idx in feat_indices:
            start_logit = start_logits[idx]
            end_logit = end_logits[idx]
            offsets = all_offset_mappings[idx]

            # Get top N_BEST start/end logits
            start_indexes = np.argsort(start_logit)[-N_BEST:][::-1]
            end_indexes = np.argsort(end_logit)[-N_BEST:][::-1]

            for start_idx in start_indexes:
                if offsets[start_idx] is None:
                    continue
                for end_idx in end_indexes:
                    if offsets[end_idx] is None:
                        continue
                    if end_idx < start_idx:
                        continue
                    if (end_idx - start_idx + 1) > MAX_ANSWER_LENGTH:
                        continue
                    score = start_logit[start_idx] + end_logit[end_idx]
                    if score > best_score:
                        start_char = offsets[start_idx][0]
                        end_char = offsets[end_idx][1]
                        answer = context[start_char:end_char]
                        best_answer = answer
                        best_score = score
        predictions[ex_id] = best_answer

    return predictions

# ---------- Training Function ----------
def train_model(train_features, epochs, model, desc="Training"):
    train_dataset = TrainQADataset(train_features)
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )

    # Optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE, eps=1e-8)

    # Scheduler
    steps_per_epoch = (len(train_loader) + GRAD_ACCUM_STEPS - 1) // GRAD_ACCUM_STEPS
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(WARMUP_RATIO * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    scaler = amp.GradScaler()
    model.train()

    global_step = 0
    for epoch in range(epochs):
        epoch_loss = 0.0
        accumulated_batches = 0
        optimizer.zero_grad()

        for batch in tqdm(train_loader, desc=f"{desc} Epoch {epoch+1}/{epochs}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            start_positions = batch["start_positions"].to(device)
            end_positions = batch["end_positions"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            with amp.autocast():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    start_positions=start_positions,
                    end_positions=end_positions
                )
                loss = outputs.loss / GRAD_ACCUM_STEPS

            scaler.scale(loss).backward()
            epoch_loss += loss.item() * GRAD_ACCUM_STEPS
            accumulated_batches += 1

            if accumulated_batches == GRAD_ACCUM_STEPS:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                accumulated_batches = 0
                global_step += 1

        # End of epoch: handle remaining gradients
        if accumulated_batches > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")

    return model

# ---------- Step 1: Train on train_split and validate ----------
print("===== Training on 80% split for validation metric =====")
train_features = prepare_train_features(train_split)
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME).to(device)
model = train_model(train_features, EPOCHS, model, desc="Train-split")

val_features = prepare_eval_features(val_split)
val_predictions = predict_answers(model, val_features, val_split)
val_score = compute_metric(val_predictions, val_split)
print(f"Validation Jaccard score: {val_score:.4f}")

# ---------- Step 2: Retrain on full training data ----------
print("\n===== Retraining on full training data for final submission =====")
del model, train_features, val_features, val_predictions
torch.cuda.empty_cache()

full_train_features = prepare_train_features(train_df)
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME).to(device)
model = train_model(full_train_features, EPOCHS, model, desc="Full-train")

# ---------- Predict on test set ----------
test_features = prepare_eval_features(test_df)
test_predictions = predict_answers(model, test_features, test_df)

# ---------- Create submission ----------
submission_df = pd.DataFrame({
    "id": list(test_predictions.keys()),
    "PredictionString": list(test_predictions.values())
})
os.makedirs("submission", exist_ok=True)
submission_df.to_csv("submission/submission.csv", index=False)
print("Submission saved to submission/submission.csv")