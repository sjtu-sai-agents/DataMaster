import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.optim import AdamW
import random
import os
import re
from tqdm import tqdm
import warnings
import csv
import sys

warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Paths
input_dir = "./input"
working_dir = "./working"
submission_dir = "./submission"

# Create directories if they don't exist
os.makedirs(working_dir, exist_ok=True)
os.makedirs(submission_dir, exist_ok=True)

# Constants
MAX_LENGTH = 128
TRAIN_BATCH_SIZE = 32
VAL_BATCH_SIZE = 64
TRAIN_STEPS = 20000
VAL_STEPS = 500
LEARNING_RATE = 3e-4


class SentenceDataset(Dataset):
    def __init__(self, sentences, tokenizer, max_length=128, train=True):
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.train = train

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx].strip()

        if self.train:
            # For training: create synthetic missing word examples
            words = sentence.split()
            if len(words) > 3:
                # Remove a random word (not first or last)
                remove_idx = random.randint(1, len(words) - 2)
                missing_word = words[remove_idx]
                masked_words = words[:remove_idx] + ["[MASK]"] + words[remove_idx + 1 :]
                masked_sentence = " ".join(masked_words)

                input_text = f"fill in the blank: {masked_sentence}"
                target_text = missing_word
            else:
                input_text = f"fill in the blank: {sentence}"
                target_text = ""
        else:
            # For test: sentences already have placeholder underscore
            input_text = f"fill in the blank: {sentence}"
            target_text = ""

        # Tokenize inputs
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        if self.train:
            target_encoding = self.tokenizer(
                target_text,
                max_length=32,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            labels = target_encoding["input_ids"]
            labels[labels == self.tokenizer.pad_token_id] = -100

            return {
                "input_ids": input_encoding["input_ids"].flatten(),
                "attention_mask": input_encoding["attention_mask"].flatten(),
                "labels": labels.flatten(),
            }
        else:
            return {
                "input_ids": input_encoding["input_ids"].flatten(),
                "attention_mask": input_encoding["attention_mask"].flatten(),
                "original_sentence": sentence,
            }


def load_data():
    """Load training and test data"""
    print("Loading data...")

    # Load training data
    train_path = os.path.join(input_dir, "train_v2.txt")
    with open(train_path, "r", encoding="utf-8") as f:
        train_sentences = f.readlines()

    # Load test data - read as CSV with proper parsing
    test_path = os.path.join(input_dir, "test_v2.txt")
    test_ids = []
    test_sentences = []

    # Read the test file line by line and parse as CSV
    with open(test_path, "r", encoding="utf-8") as f:
        # Use csv reader to properly handle quoted fields
        reader = csv.reader(f)
        header_skipped = False

        for row in reader:
            # Skip header if present
            if (
                not header_skipped
                and len(row) > 0
                and (row[0] == "id" or row[0].startswith('"id"'))
            ):
                header_skipped = True
                continue

            if len(row) >= 2:
                # First column is ID, second column is sentence
                test_ids.append(int(row[0]))
                # Remove surrounding quotes if present
                sentence = row[1].strip('"')
                test_sentences.append(sentence)
            elif len(row) == 1:
                # If only one column, assume it's the sentence and generate ID
                test_ids.append(len(test_ids) + 1)
                test_sentences.append(row[0].strip('"'))

    # If we didn't find IDs in CSV format, check if file is just sentences
    if len(test_ids) == 0:
        print("Test file doesn't appear to be in CSV format, reading as plain text...")
        with open(test_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            # Skip possible header
            first_line = lines[0].strip()
            if first_line.startswith('"id"') or first_line.startswith("id,"):
                lines = lines[1:]

            for i, line in enumerate(lines, 1):
                line = line.strip()
                # Try to extract sentence if CSV-like
                if "," in line and line.count('"') >= 2:
                    # CSV-like format: id,"sentence"
                    parts = line.split(",", 1)
                    if len(parts) == 2:
                        test_ids.append(int(parts[0]))
                        sentence = parts[1].strip().strip('"')
                        test_sentences.append(sentence)
                else:
                    # Plain text, just the sentence
                    test_ids.append(i)
                    test_sentences.append(line)

    print(f"Loaded {len(train_sentences)} training sentences")
    print(
        f"Loaded {len(test_sentences)} test sentences with {len(set(test_ids))} unique IDs"
    )

    # Verify test sentences have underscores
    has_underscore = sum(1 for s in test_sentences if "_" in s)
    print(f"Test sentences with underscore: {has_underscore}/{len(test_sentences)}")

    # Sample first few IDs for verification
    print(f"Sample test IDs: {test_ids[:5]}")
    print(f"Sample test sentences: {[s[:50] + '...' for s in test_sentences[:2]]}")

    return train_sentences, test_ids, test_sentences


def create_validation_split(train_sentences, val_size=50000):
    """Create validation split from training data"""
    total_samples = len(train_sentences)

    # Use first val_size for validation, next 1M for training
    val_sentences = train_sentences[:val_size]
    train_sentences_subset = train_sentences[val_size : val_size + 1000000]

    return train_sentences_subset, val_sentences


def train_model(train_sentences, val_sentences):
    """Train T5 model on the training data"""
    print("Initializing model and tokenizer...")

    # Initialize tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    model = model.to(device)

    # Create datasets
    train_dataset = SentenceDataset(train_sentences, tokenizer, train=True)
    val_dataset = SentenceDataset(val_sentences, tokenizer, train=True)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    print("Starting training...")
    model.train()
    train_loss = 0
    train_batches = 0

    pbar = tqdm(train_loader, desc="Training", total=TRAIN_STEPS)
    for batch in pbar:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_batches += 1
        pbar.set_postfix({"loss": loss.item()})

        if train_batches >= TRAIN_STEPS:
            break

    avg_train_loss = train_loss / train_batches

    # Validation
    print("Validating...")
    model.eval()
    val_loss = 0
    val_batches = 0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation", total=VAL_STEPS)
        for batch in pbar:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )

            val_loss += outputs.loss.item()
            val_batches += 1

            if val_batches >= VAL_STEPS:
                break

    avg_val_loss = val_loss / val_batches
    perplexity = torch.exp(torch.tensor(avg_val_loss)).item()

    print(f"\nTraining Results:")
    print(f"  Train Loss: {avg_train_loss:.4f}")
    print(f"  Val Loss: {avg_val_loss:.4f}")
    print(f"  Val Perplexity: {perplexity:.4f}")

    # Save model
    torch.save(model.state_dict(), os.path.join(working_dir, "model.pth"))

    return model, tokenizer, perplexity


def predict_missing_words(model, tokenizer, test_sentences):
    """Predict missing words for test sentences"""
    print("Generating predictions for test set...")

    test_dataset = SentenceDataset(test_sentences, tokenizer, train=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    model.eval()
    predictions = []

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Predicting")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            original_sentences = batch["original_sentence"]

            # Generate predictions
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=32,
                num_beams=5,
                early_stopping=True,
                no_repeat_ngram_size=2,
            )

            # Decode predictions
            predicted_words = tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

            # Reconstruct full sentences
            for idx, (original, pred_word) in enumerate(
                zip(original_sentences, predicted_words)
            ):
                original = original.strip()
                pred_word = pred_word.strip()

                # Handle empty predictions
                if not pred_word:
                    pred_word = "the"  # Default common word

                # Replace underscore placeholder with predicted word
                if "_" in original:
                    # Replace only the first underscore
                    parts = original.split("_", 1)
                    if len(parts) == 2:
                        reconstructed = parts[0] + pred_word + parts[1]
                    else:
                        reconstructed = original.replace("_", pred_word, 1)
                else:
                    # If no underscore found, try to insert at a likely position
                    words = original.split()
                    if len(words) > 2:
                        # Insert at a random position (not first or last)
                        insert_pos = random.randint(1, len(words) - 1)
                        words.insert(insert_pos, pred_word)
                        reconstructed = " ".join(words)
                    else:
                        reconstructed = original + " " + pred_word

                predictions.append(reconstructed)

    return predictions


def create_submission(test_ids, predictions):
    """Create submission CSV file with proper formatting using original test IDs"""
    print("Creating submission file...")

    if len(test_ids) != len(predictions):
        print(
            f"Warning: Test IDs count ({len(test_ids)}) doesn't match predictions count ({len(predictions)})"
        )
        # Use the minimum of the two
        min_len = min(len(test_ids), len(predictions))
        test_ids = test_ids[:min_len]
        predictions = predictions[:min_len]

    # Create submission DataFrame
    submission_data = []
    for test_id, sentence in zip(test_ids, predictions):
        submission_data.append({"id": test_id, "sentence": sentence})

    df = pd.DataFrame(submission_data)

    # Ensure proper CSV formatting with quotes
    submission_path = os.path.join(submission_dir, "submission.csv")

    # Use csv module for proper quote handling
    with open(submission_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["id", "sentence"])
        for _, row in df.iterrows():
            writer.writerow([int(row["id"]), row["sentence"]])

    print(f"Submission saved to {submission_path}")
    print(f"Number of predictions: {len(df)}")
    print(f"ID range: {df['id'].min()} to {df['id'].max()}")

    # Verify file was created correctly
    with open(submission_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        print(f"Submission file has {len(lines)} lines (including header)")

    return df


def compute_validation_metric(model, tokenizer, val_sentences):
    """Compute validation perplexity on a subset of validation data"""
    print("Computing validation metric...")

    val_dataset = SentenceDataset(val_sentences[:10000], tokenizer, train=True)
    val_loader = DataLoader(
        val_dataset, batch_size=VAL_BATCH_SIZE, shuffle=False, num_workers=8
    )

    model.eval()
    total_loss = 0
    total_batches = 0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Computing metric")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )

            total_loss += outputs.loss.item()
            total_batches += 1

            if total_batches >= 100:
                break

    avg_loss = total_loss / total_batches
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return perplexity


def validate_submission_format(test_ids):
    """Validate the submission file format"""
    submission_path = os.path.join(submission_dir, "submission.csv")
    if not os.path.exists(submission_path):
        print("ERROR: submission.csv not found!")
        return False

    try:
        # Read with proper CSV parsing
        df = pd.read_csv(submission_path, quoting=1)
        print(f"Submission shape: {df.shape}")

        # Check required columns
        if "id" not in df.columns or "sentence" not in df.columns:
            print("ERROR: Missing required columns")
            return False

        # Check that IDs match the test IDs
        submission_ids = df["id"].tolist()

        # Sort both lists for comparison
        sorted_submission_ids = sorted(submission_ids)
        sorted_test_ids = sorted(test_ids)

        if sorted_submission_ids != sorted_test_ids:
            print(f"ERROR: IDs don't match test IDs")
            print(
                f"Submission IDs count: {len(submission_ids)}, Test IDs count: {len(test_ids)}"
            )
            print(f"Submission IDs sample: {submission_ids[:5]}")
            print(f"Test IDs sample: {test_ids[:5]}")

            # Check if at least the counts match
            if len(submission_ids) != len(test_ids):
                print(f"Count mismatch: {len(submission_ids)} vs {len(test_ids)}")
            return False

        # Check for missing values
        if df["sentence"].isnull().any():
            print("ERROR: Missing sentences")
            return False

        print("Submission format validation passed!")
        return True

    except Exception as e:
        print(f"ERROR validating submission: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    try:
        # Load data - now returns test_ids as well
        train_sentences, test_ids, test_sentences = load_data()

        # Create validation split
        train_subset, val_sentences = create_validation_split(train_sentences)

        print(f"\nData split:")
        print(f"  Training: {len(train_subset)} sentences")
        print(f"  Validation: {len(val_sentences)} sentences")
        print(f"  Test: {len(test_sentences)} sentences")

        # Train model
        model, tokenizer, val_perplexity = train_model(train_subset, val_sentences)

        # Generate predictions
        predictions = predict_missing_words(model, tokenizer, test_sentences)

        # Create submission with original test IDs
        submission_df = create_submission(test_ids, predictions)

        # Compute final validation metric
        final_perplexity = compute_validation_metric(model, tokenizer, val_sentences)

        # Validate submission format
        format_ok = validate_submission_format(test_ids)

        # Print evaluation metric
        print(f"\n{'='*50}")
        print(f"EVALUATION METRIC (Validation Perplexity): {final_perplexity:.4f}")
        print(f"Submission format valid: {format_ok}")
        print(f"{'='*50}")

        # Display sample predictions
        print("\nSample predictions (first 5):")
        for i in range(min(5, len(predictions))):
            print(f"  ID {test_ids[i]}: {predictions[i][:100]}...")

        print(f"\nSubmission file created at: ./submission/submission.csv")
        print(f"Validation Perplexity: {final_perplexity:.4f}")

    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
