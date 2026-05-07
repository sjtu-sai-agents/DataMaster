import os
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.metrics import log_loss
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Spooky Author Identification Training')
    
    # Model parameters
    parser.add_argument('--model_name', type=str, default='microsoft/deberta-v3-large',
                        help='Pretrained model name or path')
    parser.add_argument('--max_length', type=int, default=192,
                        help='Maximum sequence length for tokenization')
    
    # Training parameters
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--train_batch_size', type=int, default=8,
                        help='Training batch size per device')
    parser.add_argument('--eval_batch_size', type=int, default=16,
                        help='Evaluation batch size per device')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                        help='Gradient accumulation steps')
    parser.add_argument('--num_epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                        help='Warmup ratio')
    parser.add_argument('--rdrop_alpha', type=float, default=4.0,
                        help='R-Drop regularization coefficient')
    
    # Seed parameters
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44],
                        help='Random seeds for ensemble')
    
    # Path parameters
    parser.add_argument('--output_dir', type=str, default='./working',
                        help='Output directory for models and logs')
    parser.add_argument('--submission_dir', type=str, default='submission',
                        help='Directory for submission files')
    
    # Other parameters
    parser.add_argument('--fp16', action='store_true', default=True,
                        help='Use mixed precision training')
    parser.add_argument('--dataloader_num_workers', type=int, default=4,
                        help='Number of dataloader workers')
    parser.add_argument('--logging_steps', type=int, default=50,
                        help='Logging steps')
    
    return parser.parse_args()


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


class SpookyDataset(Dataset):
    """Dataset for Spooky Author Identification."""
    
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


class RDropTrainer(Trainer):
    """Custom Trainer with R-Drop regularization."""
    
    def __init__(self, alpha=4.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

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


def main():
    """Main training function."""
    args = parse_args()
    
    # Set tokenizer parallelism to avoid warning
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Load data using MyDataLoader
    data_loader = MyDataLoader()
    train_data, test_data = data_loader.get_data()
    
    # Extract data
    X_train = train_data['X_train']
    y_train = train_data['y_train']
    X_val = train_data['X_val']
    y_val = train_data['y_val']
    X_holdout = train_data['X_holdout']
    y_holdout = train_data['y_holdout']
    num_labels = train_data['num_labels']
    
    X_test = test_data['X_test']
    test_ids = test_data['test_ids']
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Holdout samples: {len(X_holdout)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Number of labels: {num_labels}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Tokenize all sets
    def tokenize(texts):
        return tokenizer(
            list(texts),
            max_length=args.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
    
    train_enc = tokenize(X_train)
    val_enc = tokenize(X_val)
    holdout_enc = tokenize(X_holdout)
    test_enc = tokenize(X_test)
    
    # Create datasets
    train_dataset = SpookyDataset(train_enc, y_train)
    val_dataset = SpookyDataset(val_enc, y_val)
    holdout_dataset = SpookyDataset(holdout_enc, y_holdout)
    test_dataset = SpookyDataset(test_enc)
    
    # Lists to store predictions
    holdout_preds = []
    test_preds = []
    
    for seed in args.seeds:
        print(f"\n===== Training model with seed {seed} =====")
        set_seed(seed)

        # Create output directories
        output_dir = f"{args.output_dir}/model_seed{seed}"
        logging_dir = f"{args.output_dir}/logs_seed{seed}"
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(logging_dir, exist_ok=True)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.train_batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_train_epochs=args.num_epochs,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            fp16=args.fp16,
            logging_dir=logging_dir,
            logging_steps=args.logging_steps,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            seed=seed,
            dataloader_num_workers=args.dataloader_num_workers,
            remove_unused_columns=False,
            report_to="none",
        )

        # Load model
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name,
            num_labels=num_labels
        )

        # Initialize trainer
        trainer = RDropTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            alpha=args.rdrop_alpha,
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
    os.makedirs(args.submission_dir, exist_ok=True)

    # Save submission file
    submission_df = pd.DataFrame({
        "id": test_ids,
        "EAP": test_ensemble[:, 0],
        "HPL": test_ensemble[:, 1],
        "MWS": test_ensemble[:, 2]
    })
    submission_df.to_csv(f"{args.submission_dir}/submission.csv", index=False)
    print(f"Submission saved to {args.submission_dir}/submission.csv")


if __name__ == "__main__":
    main()