import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import os

# Set random seeds for reproducibility
np.random.seed(42)

# Load data
train_path = "./input/train.csv"
test_path = "./input/test.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# Prepare features and labels
# The first column is 'Insult' (label), second is 'Date', third is 'Comment'
# Handle column naming robustly
if 'Insult' in train_df.columns:
    y = train_df['Insult'].to_numpy()  # Convert to numpy array
    X_text = train_df['Comment'].fillna('').astype(str).to_numpy()
else:
    # If column names are different, assume first column is label, third is text
    y = train_df.iloc[:, 0].to_numpy()
    X_text = train_df.iloc[:, 2].fillna('').astype(str).to_numpy()

# Split into training and validation sets (80/20)
X_train_text, X_val_text, y_train, y_val = train_test_split(
    X_text, y, test_size=0.2, random_state=42, stratify=y
)

# TF-IDF vectorization
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words='english',
    ngram_range=(1, 2),
    min_df=2
)
X_train = vectorizer.fit_transform(X_train_text)
X_val = vectorizer.transform(X_val_text)

# Train logistic regression
model = LogisticRegression(
    C=1.0,
    solver='liblinear',
    random_state=42,
    max_iter=1000
)
model.fit(X_train, y_train)

# Predict on validation set
val_preds = model.predict_proba(X_val)[:, 1]
val_auc = roc_auc_score(y_val, val_preds)
print(f"Validation AUC: {val_auc:.4f}")

# Prepare test data
if 'Comment' in test_df.columns:
    test_text = test_df['Comment'].fillna('').astype(str).to_numpy()
    test_date = test_df['Date'].fillna('').astype(str).to_numpy()
    test_comment = test_df['Comment'].fillna('').astype(str).to_numpy()
else:
    # Assume columns: first is dummy label (all zeros), second is Date, third is Comment
    test_text = test_df.iloc[:, 2].fillna('').astype(str).to_numpy()
    test_date = test_df.iloc[:, 1].fillna('').astype(str).to_numpy()
    test_comment = test_df.iloc[:, 2].fillna('').astype(str).to_numpy()

X_test = vectorizer.transform(test_text)
test_preds = model.predict_proba(X_test)[:, 1]

# Create submission dataframe with required columns: Insult, Date, Comment
submission_df = pd.DataFrame({
    'Insult': test_preds,
    'Date': test_date,
    'Comment': test_comment
})

# Save submission
os.makedirs("./submission", exist_ok=True)
submission_path = "./submission/submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path} with shape {submission_df.shape}")