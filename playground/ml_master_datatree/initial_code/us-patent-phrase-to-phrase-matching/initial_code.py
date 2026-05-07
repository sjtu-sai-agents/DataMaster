import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from scipy import stats
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from sentence_transformers import CrossEncoder, InputExample

def main():
    # Set seeds for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Disable wandb and set tokenizer parallelism
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Paths
    train_path = "./input/train.csv"
    test_path = "./input/test.csv"
    submission_path = "./submission/submission.csv"
    working_dir = "./working"
    os.makedirs(working_dir, exist_ok=True)
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)

    # Hyperparameters
    model_name = "microsoft/deberta-v3-large"
    num_labels = 1
    num_folds = 3
    epochs = 5
    batch_size = 32
    warmup_steps = 200
    max_length = 512
    use_amp = True

    # Load data
    print("Loading data...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Preprocess: create text pairs
    def preprocess(df):
        text1 = df['anchor'] + " [CONTEXT] " + df['context']
        text2 = df['target']
        return text1, text2

    train_text1, train_text2 = preprocess(train_df)
    train_scores = train_df['score'].values.astype(float)
    test_text1, test_text2 = preprocess(test_df)
    test_ids = test_df['id'].values

    # Prepare training examples
    train_examples = []
    for i in range(len(train_df)):
        train_examples.append(InputExample(
            texts=[train_text1.iloc[i], train_text2.iloc[i]],
            label=train_scores[i]
        ))

    # Prepare test pairs for prediction
    test_pairs = list(zip(test_text1, test_text2))

    # Set up cross-validation
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)

    # Arrays to collect out-of-fold validation predictions
    all_val_preds = np.zeros(len(train_df))
    all_val_true = np.zeros(len(train_df))

    # List to collect test predictions from each fold
    test_preds_list = []

    print(f"Starting {num_folds}-fold cross-validation with model {model_name}")
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_examples)):
        print(f"\n----------- Fold {fold+1}/{num_folds} -----------")
        # Split examples
        train_examples_fold = [train_examples[i] for i in train_idx]
        val_examples_fold = [train_examples[i] for i in val_idx]

        # Create DataLoader for training
        train_dataloader = DataLoader(
            train_examples_fold,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=False          # FIX: set to False to avoid pinning non‑tensor data
        )

        # Initialize model
        model = CrossEncoder(
            model_name,
            num_labels=num_labels,
            max_length=max_length,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        # Define output path for this fold
        output_path = os.path.join(working_dir, f"cross_encoder_large_model_fold_{fold+1}")

        # Train
        print(f"Training for {epochs} epochs...")
        model.fit(
            train_dataloader=train_dataloader,
            loss_fct=nn.MSELoss(),
            epochs=epochs,
            warmup_steps=warmup_steps,
            use_amp=use_amp,
            output_path=output_path,
            show_progress_bar=True
        )

        # Validation predictions
        val_texts = [(ex.texts[0], ex.texts[1]) for ex in val_examples_fold]
        val_preds_fold = model.predict(
            val_texts,
            batch_size=batch_size * 2,
            show_progress_bar=True
        )
        all_val_preds[val_idx] = val_preds_fold
        all_val_true[val_idx] = [ex.label for ex in val_examples_fold]

        # Test predictions for this fold
        test_preds_fold = model.predict(
            test_pairs,
            batch_size=batch_size * 2,
            show_progress_bar=True
        )
        test_preds_list.append(test_preds_fold)

        # Clean up to free GPU memory
        del model
        torch.cuda.empty_cache()

    # Compute overall validation correlation
    corr, _ = stats.pearsonr(all_val_true, all_val_preds)
    print(f"\nOverall out-of-fold Pearson correlation: {corr:.6f}")

    # Average test predictions across folds
    test_preds = np.mean(test_preds_list, axis=0)
    # Clip to [0,1] for safety
    test_preds = np.clip(test_preds, 0, 1)

    # Create submission DataFrame
    submission_df = pd.DataFrame({
        "id": test_ids,
        "score": test_preds
    })

    # Save
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
    print("Done.")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()