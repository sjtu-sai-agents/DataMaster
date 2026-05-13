import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import warnings
import os
from tqdm import tqdm

warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)

# Load data
print("Loading data...")
train = pd.read_csv("./input/train.csv")
test = pd.read_csv("./input/test.csv")

# Separate features and target
X = train.drop(["id", "target"], axis=1).copy()
y = train["target"].copy().values
X_test = test.drop("id", axis=1).copy()
test_ids = test["id"].copy()

# Enhanced f_27 feature engineering
print("Processing f_27 feature with advanced engineering...")


def advanced_f27_engineering(df):
    df = df.copy()
    df["f_27_str"] = df["f_27"].astype(str)

    # Positional encoding
    for i in range(10):
        df[f"f_27_pos_{i}"] = df["f_27_str"].str[i].apply(lambda x: ord(x) - ord("A"))

    # Character frequency features
    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for char in chars:
        df[f"f_27_{char}_count"] = df["f_27_str"].str.count(char)

    # String statistics
    df["f_27_len"] = df["f_27_str"].str.len()
    df["f_27_unique"] = df["f_27_str"].apply(lambda x: len(set(str(x))))

    # Positional statistics
    pos_cols = [f"f_27_pos_{i}" for i in range(10)]
    pos_df = df[pos_cols]
    df["f_27_variance"] = pos_df.var(axis=1)
    df["f_27_mean"] = pos_df.mean(axis=1)
    df["f_27_std"] = pos_df.std(axis=1)
    df["f_27_min"] = pos_df.min(axis=1)
    df["f_27_max"] = pos_df.max(axis=1)
    df["f_27_range"] = df["f_27_max"] - df["f_27_min"]

    # Pattern features
    df["f_27_first_last_eq"] = (df["f_27_pos_0"] == df["f_27_pos_9"]).astype(int)
    df["f_27_is_palindrome"] = (
        (df[pos_cols] == df[pos_cols[::-1]].values).all(axis=1).astype(int)
    )

    # Drop original columns
    df.drop(["f_27", "f_27_str"], axis=1, inplace=True)
    return df


X = advanced_f27_engineering(X)
X_test = advanced_f27_engineering(X_test)

# Ensure both dataframes have same columns in same order
print("Aligning columns between train and test...")
common_cols = X.columns.intersection(X_test.columns)
X = X[common_cols]
X_test = X_test[common_cols]

# Feature interaction engineering for continuous features
print("Creating advanced feature interactions...")
continuous_features = [
    f"f_{i:02d}" for i in list(range(7)) + list(range(19, 27)) + [28]
]
continuous_features = [col for col in continuous_features if col in X.columns]

# Create polynomial features for top correlated features
corr_matrix = X[continuous_features].corr().abs()
top_pairs = []
for i in range(len(continuous_features)):
    for j in range(i + 1, len(continuous_features)):
        feat1, feat2 = continuous_features[i], continuous_features[j]
        if abs(corr_matrix.loc[feat1, feat2]) > 0.1:
            top_pairs.append((feat1, feat2))

for feat1, feat2 in top_pairs[:50]:
    X[f"{feat1}_mul_{feat2}"] = X[feat1] * X[feat2]
    X[f"{feat1}_div_{feat2}"] = X[feat1] / (X[feat2] + 1e-8)
    X_test[f"{feat1}_mul_{feat2}"] = X_test[feat1] * X_test[feat2]
    X_test[f"{feat1}_div_{feat2}"] = X_test[feat1] / (X_test[feat2] + 1e-8)

# Ensure column alignment again after creating new features
common_cols = X.columns.intersection(X_test.columns)
X = X[common_cols]
X_test = X_test[common_cols]

# Identify categorical and continuous columns
cat_cols = [f"f_{i:02d}" for i in range(7, 19)] + ["f_29", "f_30"]
cat_cols = [col for col in cat_cols if col in X.columns]
cont_cols = [col for col in X.columns if col not in cat_cols]

# Label encode categorical features
print("Label encoding categorical features...")
le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([X[col], X_test[col]], axis=0)
    le.fit(combined)
    X[col] = le.transform(X[col])
    X_test[col] = le.transform(X_test[col])
    le_dict[col] = le

# Scale continuous features using training data only
print("Scaling continuous features...")
scaler = StandardScaler()
X[cont_cols] = scaler.fit_transform(X[cont_cols])
X_test[cont_cols] = scaler.transform(X_test[cont_cols])

# Prepare data for neural network
print("Preparing data for neural network...")


class TabularDataset(Dataset):
    def __init__(self, df, cat_cols, cont_cols, target=None):
        self.cat_data = df[cat_cols].values.astype(np.int64)
        self.cont_data = df[cont_cols].values.astype(np.float32)
        self.target = target.astype(np.float32) if target is not None else None

    def __len__(self):
        return len(self.cat_data)

    def __getitem__(self, idx):
        if self.target is not None:
            return (
                torch.tensor(self.cat_data[idx], dtype=torch.long),
                torch.tensor(self.cont_data[idx], dtype=torch.float32),
                torch.tensor(self.target[idx], dtype=torch.float32),
            )
        else:
            return (
                torch.tensor(self.cat_data[idx], dtype=torch.long),
                torch.tensor(self.cont_data[idx], dtype=torch.float32),
            )


# Define neural network with attention mechanism
class TabularAttentionNN(nn.Module):
    def __init__(self, cat_dims, n_cont, hidden_dim=256, dropout=0.3):
        super().__init__()

        # Embedding layers for categorical features
        self.embeddings = nn.ModuleList(
            [nn.Embedding(dim, min(50, (dim + 1) // 2)) for dim in cat_dims]
        )

        emb_dim = sum([min(50, (dim + 1) // 2) for dim in cat_dims])
        total_dim = emb_dim + n_cont

        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            total_dim, num_heads=8, batch_first=True, dropout=dropout
        )

        # Main network
        self.network = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(self, cat_data, cont_data):
        # Process categorical features through embeddings
        cat_embedded = [emb(cat_data[:, i]) for i, emb in enumerate(self.embeddings)]
        cat_embedded = torch.cat(cat_embedded, dim=1)

        # Combine with continuous features
        x = torch.cat([cat_embedded, cont_data], dim=1)

        # Apply attention (add sequence dimension)
        x_attn = x.unsqueeze(1)
        x_attn, _ = self.attention(x_attn, x_attn, x_attn)
        x = x + x_attn.squeeze(1)  # Residual connection

        # Pass through main network
        return torch.sigmoid(self.network(x)).squeeze()


# Get categorical dimensions from LabelEncoder classes (FIXED)
cat_dims = [len(le_dict[col].classes_) for col in cat_cols]
n_cont = len(cont_cols)

print(f"Categorical dimensions: {cat_dims}")
print(f"Number of continuous features: {n_cont}")

# Set up cross-validation
n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

print(f"Starting {n_folds}-fold cross-validation with neural network...")
fold_scores = []
test_preds = np.zeros(len(X_test))
oof_preds = np.zeros(len(X))
models = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    print(f"\nFold {fold}/{n_folds}")

    # Split data
    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
    y_train_fold, y_val_fold = y[train_idx], y[val_idx]

    # Create datasets and dataloaders
    train_dataset = TabularDataset(X_train_fold, cat_cols, cont_cols, y_train_fold)
    val_dataset = TabularDataset(X_val_fold, cat_cols, cont_cols, y_val_fold)

    train_loader = DataLoader(
        train_dataset, batch_size=1024, shuffle=True, num_workers=8, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=2048, shuffle=False, num_workers=8, pin_memory=True
    )

    # Initialize model
    model = TabularAttentionNN(cat_dims, n_cont).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=3, factor=0.5
    )
    criterion = nn.BCELoss()

    # Training loop
    best_val_auc = 0
    patience_counter = 0
    max_patience = 10

    for epoch in range(30):
        model.train()
        train_loss = 0
        train_preds = []
        train_targets = []

        for cat_batch, cont_batch, target_batch in tqdm(
            train_loader, desc=f"Epoch {epoch+1}", leave=False
        ):
            cat_batch, cont_batch, target_batch = (
                cat_batch.to(device),
                cont_batch.to(device),
                target_batch.to(device),
            )

            optimizer.zero_grad()
            preds = model(cat_batch, cont_batch)
            loss = criterion(preds, target_batch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            train_preds.extend(preds.detach().cpu().numpy())
            train_targets.extend(target_batch.cpu().numpy())

        # Validation
        model.eval()
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for cat_batch, cont_batch, target_batch in val_loader:
                cat_batch, cont_batch, target_batch = (
                    cat_batch.to(device),
                    cont_batch.to(device),
                    target_batch.to(device),
                )
                preds = model(cat_batch, cont_batch)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(target_batch.cpu().numpy())

        train_auc = roc_auc_score(train_targets, train_preds)
        val_auc = roc_auc_score(val_targets, val_preds)

        scheduler.step(val_auc)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), f"./working/model_fold_{fold}.pt")
        else:
            patience_counter += 1

        if patience_counter >= max_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Load best model for this fold
    model.load_state_dict(torch.load(f"./working/model_fold_{fold}.pt"))
    models.append(model)

    # Make validation predictions
    model.eval()
    val_preds = []
    with torch.no_grad():
        for cat_batch, cont_batch, _ in DataLoader(
            val_dataset, batch_size=2048, shuffle=False, num_workers=8
        ):
            cat_batch, cont_batch = cat_batch.to(device), cont_batch.to(device)
            preds = model(cat_batch, cont_batch)
            val_preds.extend(preds.cpu().numpy())

    oof_preds[val_idx] = val_preds
    fold_auc = roc_auc_score(y_val_fold, val_preds)
    fold_scores.append(fold_auc)
    print(f"Fold {fold} AUC: {fold_auc:.6f}")

    # Make test predictions for this fold
    test_dataset = TabularDataset(X_test, cat_cols, cont_cols)
    test_preds_fold = []
    with torch.no_grad():
        for cat_batch, cont_batch in DataLoader(
            test_dataset, batch_size=2048, shuffle=False, num_workers=8
        ):
            cat_batch, cont_batch = cat_batch.to(device), cont_batch.to(device)
            preds = model(cat_batch, cont_batch)
            test_preds_fold.extend(preds.cpu().numpy())

    test_preds += np.array(test_preds_fold) / n_folds

# Overall validation score
overall_auc = roc_auc_score(y, oof_preds)
print(f"\n{'='*50}")
print(f"Cross-Validation Results:")
print(f"Fold AUC scores: {[f'{score:.6f}' for score in fold_scores]}")
print(f"Mean Fold AUC: {np.mean(fold_scores):.6f}")
print(f"Std Fold AUC: {np.std(fold_scores):.6f}")
print(f"Overall OOF AUC: {overall_auc:.6f}")

# Train final model on full data
print("\nTraining final model on full dataset...")
full_dataset = TabularDataset(X, cat_cols, cont_cols, y)
full_loader = DataLoader(full_dataset, batch_size=1024, shuffle=True, num_workers=8)

final_model = TabularAttentionNN(cat_dims, n_cont).to(device)
optimizer = optim.AdamW(final_model.parameters(), lr=0.001, weight_decay=1e-5)
criterion = nn.BCELoss()

for epoch in range(20):
    final_model.train()
    for cat_batch, cont_batch, target_batch in tqdm(
        full_loader, desc=f"Final Training Epoch {epoch+1}", leave=False
    ):
        cat_batch, cont_batch, target_batch = (
            cat_batch.to(device),
            cont_batch.to(device),
            target_batch.to(device),
        )

        optimizer.zero_grad()
        preds = final_model(cat_batch, cont_batch)
        loss = criterion(preds, target_batch)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(final_model.parameters(), 1.0)
        optimizer.step()

# Create ensemble predictions with final model
print("\nCreating ensemble predictions...")
final_test_preds = []
test_dataset = TabularDataset(X_test, cat_cols, cont_cols)
with torch.no_grad():
    for cat_batch, cont_batch in DataLoader(
        test_dataset, batch_size=2048, shuffle=False, num_workers=8
    ):
        cat_batch, cont_batch = cat_batch.to(device), cont_batch.to(device)
        preds = final_model(cat_batch, cont_batch)
        final_test_preds.extend(preds.cpu().numpy())

# Weighted ensemble of CV models and final model
test_preds_ensemble = test_preds * 0.7 + np.array(final_test_preds) * 0.3

# Clip predictions to valid probability range [0, 1]
test_preds_ensemble = np.clip(test_preds_ensemble, 0, 1)

# Create submission file
os.makedirs("./submission", exist_ok=True)
submission = pd.DataFrame({"id": test_ids, "target": test_preds_ensemble})
submission_path = "./submission/submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
print(f"Submission shape: {submission.shape}")

# Verify submission format matches sample
sample_submission = pd.read_csv("./input/sample_submission.csv")
print(f"\nVerifying submission format...")
print(f"Sample submission columns: {sample_submission.columns.tolist()}")
print(f"Our submission columns: {submission.columns.tolist()}")
print(f"Column match: {list(submission.columns) == list(sample_submission.columns)}")
print(
    f"ID range match: {submission['id'].min() == sample_submission['id'].min()} and {submission['id'].max() == sample_submission['id'].max()}"
)
print(
    f"Target value range: [{submission['target'].min():.6f}, {submission['target'].max():.6f}]"
)

# Check if file exists and is valid
if os.path.exists(submission_path):
    print(
        f"\n✓ Submission file successfully created at {os.path.abspath(submission_path)}"
    )
    submission_check = pd.read_csv(submission_path)
    print(f"✓ Submission file contains {len(submission_check)} rows")
    print(f"✓ First few rows of submission:")
    print(submission_check.head())
else:
    print(f"\n✗ ERROR: Submission file not found at {submission_path}")

# Clean up temporary files
for fold in range(1, n_folds + 1):
    if os.path.exists(f"./working/model_fold_{fold}.pt"):
        os.remove(f"./working/model_fold_{fold}.pt")

print("\nDone!")
print(f"Evaluation metric (OOF AUC): {overall_auc:.6f}")
