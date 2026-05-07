import os
import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from scipy.sparse import hstack
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import textstat

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

def extract_text_features(text):
    """Extract 9 advanced text features from a string."""
    if not text or pd.isna(text):
        return [0.0] * 9
    try:
        sent = analyzer.polarity_scores(text)
    except:
        sent = {'compound': 0.0, 'pos': 0.0, 'neu': 0.0, 'neg': 0.0}
    try:
        flesch = textstat.flesch_reading_ease(text)
        kincaid = textstat.flesch_kincaid_grade(text)
    except:
        flesch = 0.0
        kincaid = 0.0
    excl = text.count('!')
    quest = text.count('?')
    if len(text) > 0:
        upper_ratio = sum(1 for c in text if c.isupper()) / len(text)
    else:
        upper_ratio = 0.0
    return [sent['compound'], sent['pos'], sent['neu'], sent['neg'],
            flesch, kincaid, excl, quest, upper_ratio]

# Ensure submission directory exists
os.makedirs('./submission', exist_ok=True)

# ---------- Load data ----------
print("Loading data...")
with open('./input/train/train.json', 'r') as f:
    train = pd.DataFrame(json.load(f))
with open('./input/test/test.json', 'r') as f:
    test = pd.DataFrame(json.load(f))

# Target
y = train['requester_received_pizza'].astype(int)

# ---------- Combine text ----------
train['combined_text'] = train['request_title'].fillna('') + ' ' + train['request_text_edit_aware'].fillna('')
test['combined_text'] = test['request_title'].fillna('') + ' ' + test['request_text_edit_aware'].fillna('')

# ---------- Advanced text features ----------
adv_text_cols = ['sent_compound', 'sent_pos', 'sent_neu', 'sent_neg',
                 'flesch', 'kincaid', 'excl', 'quest', 'upper_ratio']

print("Extracting advanced text features for train...")
train_adv = train['combined_text'].apply(extract_text_features).apply(pd.Series)
train_adv.columns = adv_text_cols
train = pd.concat([train, train_adv], axis=1)

print("Extracting advanced text features for test...")
test_adv = test['combined_text'].apply(extract_text_features).apply(pd.Series)
test_adv.columns = adv_text_cols
test = pd.concat([test, test_adv], axis=1)

# ---------- Basic meta features ----------
# Giver known indicator
train['giver_known'] = train['giver_username_if_known'].apply(lambda x: 0 if x == "N/A" else 1)
test['giver_known'] = test['giver_username_if_known'].apply(lambda x: 0 if x == "N/A" else 1)

# Time features from UTC timestamp
train['req_time'] = pd.to_datetime(train['unix_timestamp_of_request_utc'], unit='s', errors='coerce')
test['req_time'] = pd.to_datetime(test['unix_timestamp_of_request_utc'], unit='s', errors='coerce')
train['request_hour'] = train['req_time'].dt.hour.fillna(0).astype(int)
train['request_dayofweek'] = train['req_time'].dt.dayofweek.fillna(0).astype(int)
test['request_hour'] = test['req_time'].dt.hour.fillna(0).astype(int)
test['request_dayofweek'] = test['req_time'].dt.dayofweek.fillna(0).astype(int)
train.drop('req_time', axis=1, inplace=True)
test.drop('req_time', axis=1, inplace=True)

# ---------- Length features ----------
train['text_length'] = train['request_text_edit_aware'].fillna('').apply(len)
test['text_length'] = test['request_text_edit_aware'].fillna('').apply(len)
train['title_length'] = train['request_title'].fillna('').apply(len)
test['title_length'] = test['request_title'].fillna('').apply(len)

# ---------- Behavioral ratio features ----------
eps = 1e-6
train['comments_per_day'] = train['requester_number_of_comments_at_request'] / (train['requester_account_age_in_days_at_request'] + eps)
test['comments_per_day'] = test['requester_number_of_comments_at_request'] / (test['requester_account_age_in_days_at_request'] + eps)

train['posts_per_day'] = train['requester_number_of_posts_at_request'] / (train['requester_account_age_in_days_at_request'] + eps)
test['posts_per_day'] = test['requester_number_of_posts_at_request'] / (test['requester_account_age_in_days_at_request'] + eps)

train['raop_comments_ratio'] = train['requester_number_of_comments_in_raop_at_request'] / (train['requester_number_of_comments_at_request'] + eps)
test['raop_comments_ratio'] = test['requester_number_of_comments_in_raop_at_request'] / (test['requester_number_of_comments_at_request'] + eps)

train['raop_posts_ratio'] = train['requester_number_of_posts_on_raop_at_request'] / (train['requester_number_of_posts_at_request'] + eps)
test['raop_posts_ratio'] = test['requester_number_of_posts_on_raop_at_request'] / (test['requester_number_of_posts_at_request'] + eps)

train['upvotes_per_comment'] = train['requester_upvotes_plus_downvotes_at_request'] / (train['requester_number_of_comments_at_request'] + eps)
test['upvotes_per_comment'] = test['requester_upvotes_plus_downvotes_at_request'] / (test['requester_number_of_comments_at_request'] + eps)

train['upvotes_per_post'] = train['requester_upvotes_plus_downvotes_at_request'] / (train['requester_number_of_posts_at_request'] + eps)
test['upvotes_per_post'] = test['requester_upvotes_plus_downvotes_at_request'] / (test['requester_number_of_posts_at_request'] + eps)

train['subreddits_per_day'] = train['requester_number_of_subreddits_at_request'] / (train['requester_account_age_in_days_at_request'] + eps)
test['subreddits_per_day'] = test['requester_number_of_subreddits_at_request'] / (test['requester_account_age_in_days_at_request'] + eps)

train['text_to_title_ratio'] = train['text_length'] / (train['title_length'] + eps)
test['text_to_title_ratio'] = test['text_length'] / (test['title_length'] + eps)

# ---------- Define feature sets ----------
base_meta_cols = [
    'requester_account_age_in_days_at_request',
    'requester_days_since_first_post_on_raop_at_request',
    'requester_number_of_comments_at_request',
    'requester_number_of_comments_in_raop_at_request',
    'requester_number_of_posts_at_request',
    'requester_number_of_posts_on_raop_at_request',
    'requester_number_of_subreddits_at_request',
    'requester_upvotes_minus_downvotes_at_request',
    'requester_upvotes_plus_downvotes_at_request',
    'unix_timestamp_of_request',
    'unix_timestamp_of_request_utc',
    'giver_known',
    'request_hour',
    'request_dayofweek'
]

ratio_cols = [
    'comments_per_day',
    'posts_per_day',
    'raop_comments_ratio',
    'raop_posts_ratio',
    'upvotes_per_comment',
    'upvotes_per_post',
    'text_length',
    'title_length',
    'text_to_title_ratio',
    'subreddits_per_day'
]

meta_cols = base_meta_cols + adv_text_cols + ratio_cols

# Fill missing values
for col in meta_cols:
    train[col] = train[col].fillna(0)
    test[col] = test[col].fillna(0)

# ---------- TF-IDF on combined text ----------
print("Fitting TF-IDF...")
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english',
                             ngram_range=(1,2), min_df=2, max_df=0.95)
X_train_tfidf = vectorizer.fit_transform(train['combined_text'].fillna(''))
X_test_tfidf = vectorizer.transform(test['combined_text'].fillna(''))

# ---------- Combine all features ----------
X_train_meta = train[meta_cols].values
X_test_meta = test[meta_cols].values

X_train = hstack([X_train_tfidf, X_train_meta]).tocsr()
X_test = hstack([X_test_tfidf, X_test_meta]).tocsr()

# ---------- Cross‑validation with LightGBM ----------
print("Starting cross-validation...")
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'num_threads': -1,          # use all CPU cores
    'seed': 42,
    'device': 'gpu',            # leverage the A100 GPU
}

fold_aucs = []
test_preds = np.zeros(len(test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y)):
    print(f"Fold {fold+1}")
    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    lgb_train = lgb.Dataset(X_tr, label=y_tr)
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

    pos = y_tr.sum()
    neg = len(y_tr) - pos
    scale_pos_weight = neg / pos if pos > 0 else 1.0
    params_fold = params.copy()
    params_fold['scale_pos_weight'] = scale_pos_weight

    model = lgb.train(params_fold, lgb_train, num_boost_round=1000,
                      valid_sets=[lgb_val],
                      callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])

    val_pred = model.predict(X_val)
    auc = roc_auc_score(y_val, val_pred)
    fold_aucs.append(auc)
    print(f"Fold {fold+1} AUC: {auc:.5f}")

    test_preds += model.predict(X_test)

test_preds /= n_splits

cv_mean = np.mean(fold_aucs)
cv_std = np.std(fold_aucs)
print(f"CV AUC: {cv_mean:.5f} ± {cv_std:.5f}")

# ---------- Save submission ----------
submission = pd.DataFrame({
    'request_id': test['request_id'],
    'requester_received_pizza': test_preds
})
submission.to_csv('./submission/submission.csv', index=False)
print("Submission saved to ./submission/submission.csv")