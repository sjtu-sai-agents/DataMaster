import os
import random
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import make_scorer
from scipy.stats import spearmanr

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
os.environ['PYTHONHASHSEED'] = '42'

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def safe(s):
    return str(s) if pd.notnull(s) else ""

def build_question_text(row):
    title = safe(row['question_title'])
    body = safe(row['question_body'])
    category = safe(row['category'])
    host = safe(row['host'])
    return f"title: {title} [SEP] body: {body} [SEP] category: {category} [SEP] host: {host}".lower()

def build_answer_text(row):
    answer = safe(row['answer'])
    category = safe(row['category'])
    host = safe(row['host'])
    return f"answer: {answer} [SEP] category: {category} [SEP] host: {host}".lower()

# ------------------------------------------------------------
# Data loading
# ------------------------------------------------------------
train = pd.read_csv('./input/train.csv')
test = pd.read_csv('./input/test.csv')
submission_template = pd.read_csv('./input/sample_submission.csv')
target_columns = [col for col in submission_template.columns if col != 'qa_id']

# ------------------------------------------------------------
# Build text fields
# ------------------------------------------------------------
print("Building texts...")
train_q_texts = train.apply(build_question_text, axis=1).tolist()
train_a_texts = train.apply(build_answer_text, axis=1).tolist()
test_q_texts = test.apply(build_question_text, axis=1).tolist()
test_a_texts = test.apply(build_answer_text, axis=1).tolist()

# ------------------------------------------------------------
# TF-IDF feature extraction
# ------------------------------------------------------------
def build_tfidf_features(train_texts, test_texts, all_texts):
    word_vec = TfidfVectorizer(analyzer='word', ngram_range=(1,2), max_features=50000,
                               sublinear_tf=True, lowercase=True, strip_accents='unicode')
    char_vec = TfidfVectorizer(analyzer='char', ngram_range=(3,5), max_features=50000,
                               sublinear_tf=True, lowercase=True, strip_accents='unicode')
    # fit on all texts (train+test)
    word_vec.fit(all_texts)
    char_vec.fit(all_texts)
    X_train_word = word_vec.transform(train_texts)
    X_train_char = char_vec.transform(train_texts)
    X_test_word = word_vec.transform(test_texts)
    X_test_char = char_vec.transform(test_texts)
    X_train = hstack([X_train_word, X_train_char], format='csr')
    X_test = hstack([X_test_word, X_test_char], format='csr')
    return X_train, X_test

print("Building question features...")
X_q_train, X_q_test = build_tfidf_features(train_q_texts, test_q_texts, train_q_texts+test_q_texts)
print("Building answer features...")
X_a_train, X_a_test = build_tfidf_features(train_a_texts, test_a_texts, train_a_texts+test_a_texts)

# ------------------------------------------------------------
# Target values
# ------------------------------------------------------------
y_all = train[target_columns].values.astype(np.float32)

# ------------------------------------------------------------
# Train / validation split (90% / 10%)
# ------------------------------------------------------------
n_train = len(train)
indices = np.arange(n_train)
train_idx, val_idx = train_test_split(indices, test_size=0.1, random_state=42)

X_q_train_sub = X_q_train[train_idx]
X_a_train_sub = X_a_train[train_idx]
X_q_val = X_q_train[val_idx]
X_a_val = X_a_train[val_idx]
y_train = y_all[train_idx]
y_val = y_all[val_idx]

# ------------------------------------------------------------
# Hyperparameter grids
# ------------------------------------------------------------
lambdas = [0.0, 0.25, 0.5, 1.0]
alphas = [0.3, 0.7, 1.5, 3.0, 6.0, 12.0, 24.0]

# ------------------------------------------------------------
# Pre‑compute combined matrices for each lambda
# ------------------------------------------------------------
def build_group_mats(X_q, X_a, lambdas):
    mats = {}
    for lam in lambdas:
        if lam == 0:
            mats[lam] = {'question': X_q, 'answer': X_a}
        else:
            mats[lam] = {
                'question': hstack([X_q, lam * X_a], format='csr'),
                'answer': hstack([X_a, lam * X_q], format='csr')
            }
    return mats

print("Pre‑computing combined matrices for each lambda...")
train_mats = build_group_mats(X_q_train_sub, X_a_train_sub, lambdas)
val_mats   = build_group_mats(X_q_val, X_a_val, lambdas)
full_mats  = build_group_mats(X_q_train, X_a_train, lambdas)
test_mats  = build_group_mats(X_q_test, X_a_test, lambdas)

# ------------------------------------------------------------
# Spearman scorer for cross‑validation
# ------------------------------------------------------------
def spearman_scorer_func(y_true, y_pred):
    corr = spearmanr(y_true, y_pred).correlation
    return 0.0 if np.isnan(corr) else corr

spearman_scorer = make_scorer(spearman_scorer_func, greater_is_better=True)

# ------------------------------------------------------------
# Per‑target hyperparameter tuning and validation
# ------------------------------------------------------------
best_params = {}   # col -> (alpha, lam)
val_scores = []

print("Per target hyperparameter tuning and validation...")
for i, col in enumerate(target_columns):
    print(f"Processing target {i+1}/30: {col}")
    group = 'question' if col.startswith('question') else 'answer'
    y_train_col = y_train[:, i]
    y_val_col   = y_val[:, i]

    # ---------- Grid search over (lambda, alpha) ----------
    best_score = -np.inf
    best_alpha = None
    best_lam = None
    for lam in lambdas:
        X = train_mats[lam][group]
        for alpha in alphas:
            model = Ridge(alpha=alpha, random_state=42)
            scores = cross_val_score(model, X, y_train_col,
                                     cv=KFold(5, shuffle=True, random_state=42),
                                     scoring=spearman_scorer, n_jobs=1)
            mean_score = np.mean(scores)
            if mean_score > best_score:
                best_score = mean_score
                best_alpha = alpha
                best_lam = lam
    best_params[col] = (best_alpha, best_lam)

    # ---------- OOF predictions on training subset for isotonic ----------
    X_best = train_mats[best_lam][group]
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(train_idx))
    for tr_idx, va_idx in kf.split(X_best):
        model = Ridge(alpha=best_alpha, random_state=42)
        model.fit(X_best[tr_idx], y_train_col[tr_idx])
        oof_preds[va_idx] = model.predict(X_best[va_idx])

    # Fit isotonic regression
    try:
        iso = IsotonicRegression(out_of_bounds='clip', increasing=True)
        iso.fit(oof_preds.reshape(-1,1), y_train_col)
    except:
        iso = None

    # ---------- Train final model on whole training subset ----------
    final_model = Ridge(alpha=best_alpha, random_state=42)
    final_model.fit(X_best, y_train_col)

    # ---------- Predict on validation set ----------
    X_val_best = val_mats[best_lam][group]
    raw_val_pred = final_model.predict(X_val_best)
    if iso is not None:
        val_pred = iso.predict(raw_val_pred.reshape(-1,1))
    else:
        val_pred = raw_val_pred
    val_pred = np.clip(val_pred, 0.0, 1.0)

    corr = spearmanr(val_pred, y_val_col).correlation
    if np.isnan(corr):
        corr = 0.0
    val_scores.append(corr)
    print(f"  Validation Spearman: {corr:.6f}")

# ------------------------------------------------------------
# Overall validation score
# ------------------------------------------------------------
mean_val_score = np.mean(val_scores)
print(f"\nOverall Validation Mean Spearman Correlation: {mean_val_score:.6f}")

# ------------------------------------------------------------
# Generate test predictions (final submission)
# ------------------------------------------------------------
submission_df = pd.DataFrame({'qa_id': test['qa_id']})

print("\nGenerating test predictions...")
for i, col in enumerate(target_columns):
    print(f"Target {i+1}/30: {col}")
    group = 'question' if col.startswith('question') else 'answer'
    best_alpha, best_lam = best_params[col]
    y_full_col = y_all[:, i]
    X_full_best = full_mats[best_lam][group]

    # ---------- OOF predictions on full data for isotonic ----------
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_full = np.zeros(len(y_full_col))
    for tr_idx, va_idx in kf.split(X_full_best):
        model = Ridge(alpha=best_alpha, random_state=42)
        model.fit(X_full_best[tr_idx], y_full_col[tr_idx])
        oof_full[va_idx] = model.predict(X_full_best[va_idx])

    # Fit isotonic regression on full data OOF
    try:
        iso_full = IsotonicRegression(out_of_bounds='clip', increasing=True)
        iso_full.fit(oof_full.reshape(-1,1), y_full_col)
    except:
        iso_full = None

    # ---------- Train final model on full data ----------
    final_full = Ridge(alpha=best_alpha, random_state=42)
    final_full.fit(X_full_best, y_full_col)

    # ---------- Predict test set ----------
    X_test_best = test_mats[best_lam][group]
    raw_test = final_full.predict(X_test_best)
    if iso_full is not None:
        test_pred = iso_full.predict(raw_test.reshape(-1,1))
    else:
        test_pred = raw_test
    test_pred = np.clip(test_pred, 0.0, 1.0)

    submission_df[col] = test_pred

# ------------------------------------------------------------
# Ensure correct column order and save
# ------------------------------------------------------------
submission_df = submission_df[['qa_id'] + target_columns]
os.makedirs('./submission', exist_ok=True)
submission_df.to_csv('./submission/submission.csv', index=False)
print("Submission saved to ./submission/submission.csv")