import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
import warnings

warnings.filterwarnings("ignore")

# Load training data (sample 1 million rows for efficiency)
print("Loading data...")
train = pd.read_csv("./input/train.csv", nrows=1000000)
test = pd.read_csv("./input/test.csv")


# Preprocess text: combine title and body, remove HTML tags, lowercase
def preprocess(text):
    if isinstance(text, str):
        # Remove HTML tags
        text = re.sub(r"<[^>]+>", " ", text)
        # Replace multiple spaces with single space
        text = re.sub(r"\s+", " ", text)
        return text.lower().strip()
    return ""


print("Preprocessing text...")
train["text"] = (train["Title"].fillna("") + " " + train["Body"].fillna("")).apply(
    preprocess
)
test["text"] = (test["Title"].fillna("") + " " + test["Body"].fillna("")).apply(
    preprocess
)

# Prepare tags: split string into list
train["Tags"] = (
    train["Tags"].fillna("").apply(lambda x: x.split() if isinstance(x, str) else [])
)

# Limit to top 500 tags
all_tags = [tag for tags in train["Tags"] for tag in tags]
tag_counts = pd.Series(all_tags).value_counts()
top_tags = tag_counts.head(500).index.tolist()
train["Tags"] = train["Tags"].apply(lambda x: [tag for tag in x if tag in top_tags])

# Encode tags with MultiLabelBinarizer
mlb = MultiLabelBinarizer(classes=top_tags)
y = mlb.fit_transform(train["Tags"])

# Split into train and validation sets (80-20)
X_train, X_val, y_train, y_val = train_test_split(
    train["text"], y, test_size=0.2, random_state=42
)

# TF-IDF vectorization with limited features
print("Vectorizing text...")
vectorizer = TfidfVectorizer(
    max_features=10000, stop_words="english", ngram_range=(1, 2)
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)

# Train One-vs-Rest logistic regression
print("Training model...")
clf = OneVsRestClassifier(
    LogisticRegression(solver="sag", max_iter=100, random_state=42, n_jobs=-1)
)
clf.fit(X_train_tfidf, y_train)

# Predict probabilities on validation set
y_val_pred_prob = clf.predict_proba(X_val_tfidf)

# Tune threshold to maximize Mean F1-Score
best_threshold = 0.25
best_f1 = 0
for threshold in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
    y_val_pred = (y_val_pred_prob >= threshold).astype(int)
    f1 = f1_score(y_val, y_val_pred, average="samples", zero_division=0)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"Best threshold: {best_threshold:.2f}")
print(f"Validation Mean F1-Score: {best_f1:.4f}")

# Retrain on full sampled data (1 million rows) and predict on test set
print("Retraining on full sampled data...")
X_full_tfidf = vectorizer.fit_transform(train["text"])
clf_full = OneVsRestClassifier(
    LogisticRegression(solver="sag", max_iter=100, random_state=42, n_jobs=-1)
)
clf_full.fit(X_full_tfidf, y)

# Transform test text
X_test_tfidf = vectorizer.transform(test["text"])

# Predict on test set with best threshold
y_test_pred_prob = clf_full.predict_proba(X_test_tfidf)
y_test_pred = (y_test_pred_prob >= best_threshold).astype(int)

# Convert predictions to tag strings
test_tags = mlb.inverse_transform(y_test_pred)
test["Tags"] = [" ".join(tags) for tags in test_tags]

# Ensure no ampersands or tabs in tags (as per instructions)
test["Tags"] = test["Tags"].str.replace("&", "and").str.replace("\t", " ")

# Create submission directory if not exists
import os

os.makedirs("./submission", exist_ok=True)

# Save submission file
submission = test[["Id", "Tags"]]
submission.to_csv("./submission/submission.csv", index=False)
print("Submission saved to ./submission/submission.csv")
print(f"Number of test predictions: {len(submission)}")
