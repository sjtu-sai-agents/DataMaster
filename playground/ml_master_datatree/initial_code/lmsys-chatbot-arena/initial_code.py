import os
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
import lightgbm as lgb

# Set seeds
np.random.seed(42)

# -------------------- Text cleaning --------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'^\["', '', text)   # remove leading [" 
    text = re.sub(r'"\]$', '', text)   # remove trailing "]
    text = re.sub(r'\\n', ' ', text)   # replace \n
    text = re.sub(r'\n', ' ', text)    # replace actual newline
    text = re.sub(r'\s+', ' ', text).strip()  # collapse spaces
    text = re.sub(r'\\"', '"', text)   # unescape quotes
    return text

def is_refusal(text):
    text_lower = text.lower()
    for phrase in ["sorry", "cannot", "can't", "unable to", "apologize", "language model", "ai assistant"]:
        if phrase in text_lower:
            return 1
    return 0

def jaccard_sim(text1, text2):
    set1 = set(text1.split())
    set2 = set(text2.split())
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)

# -------------------- Preprocess DataFrame --------------------
def preprocess_df(df):
    df = df.copy()
    df['clean_prompt'] = df['prompt'].apply(clean_text)
    df['clean_a'] = df['response_a'].apply(clean_text)
    df['clean_b'] = df['response_b'].apply(clean_text)
    df['len_a'] = df['clean_a'].apply(lambda x: len(x.split()))
    df['len_b'] = df['clean_b'].apply(lambda x: len(x.split()))
    df['len_diff'] = df['len_a'] - df['len_b']
    df['refusal_a'] = df['clean_a'].apply(is_refusal)
    df['refusal_b'] = df['clean_b'].apply(is_refusal)
    df['refusal_diff'] = df['refusal_a'] - df['refusal_b']
    df['jacc_p_a'] = df.apply(lambda row: jaccard_sim(row['clean_prompt'], row['clean_a']), axis=1)
    df['jacc_p_b'] = df.apply(lambda row: jaccard_sim(row['clean_prompt'], row['clean_b']), axis=1)
    df['jacc_diff'] = df['jacc_p_a'] - df['jacc_p_b']
    return df

# -------------------- Load data --------------------
train_df = pd.read_csv('./input/train.csv').fillna('')
test_df = pd.read_csv('./input/test.csv').fillna('')

print("Preprocessing data...")
train_df = preprocess_df(train_df)
test_df = preprocess_df(test_df)

# -------------------- Prepare corpora --------------------
response_corpus = list(train_df['clean_a']) + list(train_df['clean_b']) + list(test_df['clean_a']) + list(test_df['clean_b'])
prompt_corpus = list(train_df['clean_prompt']) + list(test_df['clean_prompt'])
common_corpus = response_corpus + prompt_corpus

# -------------------- TF-IDF & SVD --------------------
print("Fitting vectorizers and SVD...")
# Word-level for responses
word_vectorizer = TfidfVectorizer(
    analyzer='word',
    token_pattern=r'\w{1,}',
    ngram_range=(1,3),
    max_features=30000,
    sublinear_tf=True
)
word_tfidf = word_vectorizer.fit_transform(response_corpus)
word_svd = TruncatedSVD(n_components=64, random_state=42)
word_svd.fit(word_tfidf)

# Char-level for responses
char_vectorizer = TfidfVectorizer(
    analyzer='char',
    ngram_range=(3,5),
    max_features=30000,
    sublinear_tf=True
)
char_tfidf = char_vectorizer.fit_transform(response_corpus)
char_svd = TruncatedSVD(n_components=32, random_state=42)
char_svd.fit(char_tfidf)

# Common vectorizer for similarities (L2 normalized)
common_vectorizer = TfidfVectorizer(
    analyzer='word',
    token_pattern=r'\w{1,}',
    ngram_range=(1,3),
    max_features=20000,
    sublinear_tf=True,
    norm='l2'
)
common_vectorizer.fit(common_corpus)

# Prompt vectorizer
prompt_vectorizer = TfidfVectorizer(
    analyzer='word',
    token_pattern=r'\w{1,}',
    ngram_range=(1,3),
    max_features=15000,
    sublinear_tf=True
)
prompt_tfidf = prompt_vectorizer.fit_transform(prompt_corpus)
prompt_svd = TruncatedSVD(n_components=32, random_state=42)
prompt_svd.fit(prompt_tfidf)

# Dimensions
WORD_DIM = word_svd.n_components
CHAR_DIM = char_svd.n_components
PROMPT_DIM = prompt_svd.n_components
META_DIM = 6
SIM_DIM = 3
JACC_DIM = 3

# Slices for feature blocks (used in augmentation)
start = 0
a_word_slice = slice(start, start+WORD_DIM); start += WORD_DIM
b_word_slice = slice(start, start+WORD_DIM); start += WORD_DIM
diff_word_slice = slice(start, start+WORD_DIM); start += WORD_DIM
a_char_slice = slice(start, start+CHAR_DIM); start += CHAR_DIM
b_char_slice = slice(start, start+CHAR_DIM); start += CHAR_DIM
diff_char_slice = slice(start, start+CHAR_DIM); start += CHAR_DIM
meta_slice = slice(start, start+META_DIM); start += META_DIM
sim_slice = slice(start, start+SIM_DIM); start += SIM_DIM
jacc_slice = slice(start, start+JACC_DIM); start += JACC_DIM
prompt_slice = slice(start, start+PROMPT_DIM); start += PROMPT_DIM

# -------------------- Feature extraction --------------------
def get_features(df):
    # Word SVD
    a_word = word_svd.transform(word_vectorizer.transform(df['clean_a']))
    b_word = word_svd.transform(word_vectorizer.transform(df['clean_b']))
    diff_word = a_word - b_word

    # Char SVD
    a_char = char_svd.transform(char_vectorizer.transform(df['clean_a']))
    b_char = char_svd.transform(char_vectorizer.transform(df['clean_b']))
    diff_char = a_char - b_char

    # Meta features
    meta = df[['len_a', 'len_b', 'len_diff', 'refusal_a', 'refusal_b', 'refusal_diff']].values

    # Similarity features using common vectorizer (already L2-normalized)
    vec_p = common_vectorizer.transform(df['clean_prompt'])
    vec_a = common_vectorizer.transform(df['clean_a'])
    vec_b = common_vectorizer.transform(df['clean_b'])
    sim_a_b = np.array((vec_a.multiply(vec_b)).sum(axis=1)).ravel()
    sim_p_a = np.array((vec_p.multiply(vec_a)).sum(axis=1)).ravel()
    sim_p_b = np.array((vec_p.multiply(vec_b)).sum(axis=1)).ravel()
    sim = np.column_stack([sim_a_b, sim_p_a, sim_p_b])

    # Jaccard features
    jacc = df[['jacc_p_a', 'jacc_p_b', 'jacc_diff']].values

    # Prompt dense
    prompt_dense = prompt_svd.transform(prompt_vectorizer.transform(df['clean_prompt']))

    # Concatenate all
    features = np.hstack([
        a_word, b_word, diff_word,
        a_char, b_char, diff_char,
        meta,
        sim,
        jacc,
        prompt_dense
    ])
    return features

print("Extracting features from train...")
X_full = get_features(train_df)
print("Extracting features from test...")
X_test = get_features(test_df)

# -------------------- Target --------------------
def get_targets(df):
    # winner columns: winner_model_a, winner_model_b, winner_tie
    conditions = [
        df['winner_model_a'] == 1,
        df['winner_model_b'] == 1,
        df['winner_tie'] == 1
    ]
    choices = [0, 1, 2]
    y = np.select(conditions, choices, default=-1)
    # Should be no default
    return y.astype(int)

y_full = get_targets(train_df)

# -------------------- Augmentation function --------------------
def augment_features(X, y):
    X_aug = X.copy()
    # Swap word A and B
    X_aug[:, a_word_slice] = X[:, b_word_slice]
    X_aug[:, b_word_slice] = X[:, a_word_slice]
    X_aug[:, diff_word_slice] = -X[:, diff_word_slice]

    # Swap char A and B
    X_aug[:, a_char_slice] = X[:, b_char_slice]
    X_aug[:, b_char_slice] = X[:, a_char_slice]
    X_aug[:, diff_char_slice] = -X[:, diff_char_slice]

    # Meta: order: len_a, len_b, len_diff, refusal_a, refusal_b, refusal_diff
    meta = X[:, meta_slice]
    meta_aug = meta.copy()
    meta_aug[:, 0] = meta[:, 1]  # len_b -> len_a
    meta_aug[:, 1] = meta[:, 0]  # len_a -> len_b
    meta_aug[:, 2] = -meta[:, 2] # len_diff
    meta_aug[:, 3] = meta[:, 4]  # refusal_b -> refusal_a
    meta_aug[:, 4] = meta[:, 3]  # refusal_a -> refusal_b
    meta_aug[:, 5] = -meta[:, 5] # refusal_diff
    X_aug[:, meta_slice] = meta_aug

    # Sim: order sim_a_b, sim_p_a, sim_p_b
    sim = X[:, sim_slice]
    sim_aug = sim.copy()
    sim_aug[:, 0] = sim[:, 0]  # sim_a_b unchanged
    sim_aug[:, 1] = sim[:, 2]  # sim_p_b -> sim_p_a
    sim_aug[:, 2] = sim[:, 1]  # sim_p_a -> sim_p_b
    X_aug[:, sim_slice] = sim_aug

    # Jaccard: order jacc_p_a, jacc_p_b, jacc_diff
    jacc = X[:, jacc_slice]
    jacc_aug = jacc.copy()
    jacc_aug[:, 0] = jacc[:, 1]
    jacc_aug[:, 1] = jacc[:, 0]
    jacc_aug[:, 2] = -jacc[:, 2]
    X_aug[:, jacc_slice] = jacc_aug

    # Prompt unchanged

    # Adjust labels
    y_aug = y.copy()
    y_aug = np.where(y == 0, 1, np.where(y == 1, 0, y))

    return X_aug, y_aug

# -------------------- Cross-validation & LightGBM --------------------
N_SPLITS = 5
n_train = X_full.shape[0]
n_test = X_test.shape[0]
num_classes = 3

oof_logits = np.zeros((n_train, num_classes))
oof_labels = np.zeros(n_train, dtype=int)
test_logits = np.zeros((n_test, num_classes))

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

print("Starting cross-validation...")
for fold, (train_idx, val_idx) in enumerate(skf.split(X_full, y_full)):
    print(f"Fold {fold+1}/{N_SPLITS}")
    X_train = X_full[train_idx]
    y_train = y_full[train_idx]
    X_val = X_full[val_idx]
    y_val = y_full[val_idx]

    # Augment training data
    X_train_aug, y_train_aug = augment_features(X_train, y_train)
    X_train_combined = np.vstack([X_train, X_train_aug])
    y_train_combined = np.concatenate([y_train, y_train_aug])

    # LightGBM
    train_set = lgb.Dataset(X_train_combined, label=y_train_combined)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

    params = {
        'objective': 'multiclass',
        'num_class': num_classes,
        'metric': 'multi_logloss',
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.9,
        'bagging_freq': 1,
        'min_child_samples': 20,
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': -1
    }

    model = lgb.train(
        params,
        train_set,
        num_boost_round=1300,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )

    # Raw logits for validation and test
    val_logits = model.predict(X_val, raw_score=True)
    test_logits_fold = model.predict(X_test, raw_score=True)

    oof_logits[val_idx] = val_logits
    oof_labels[val_idx] = y_val
    test_logits += test_logits_fold

test_logits /= N_SPLITS

# -------------------- Temperature scaling --------------------
def softmax(z, temp=1.0):
    z = z / temp
    z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def eval_temp(temp, logits, labels):
    probs = softmax(logits, temp)
    return log_loss(labels, probs)

# Coarse search
coarse_temps = np.linspace(0.5, 3.0, 51)
best_temp = 1.0
best_loss = eval_temp(best_temp, oof_logits, oof_labels)
for temp in coarse_temps:
    loss = eval_temp(temp, oof_logits, oof_labels)
    if loss < best_loss:
        best_loss = loss
        best_temp = temp

# Fine search
fine_temps = np.linspace(max(0.05, best_temp-0.5), best_temp+0.5, 51)
for temp in fine_temps:
    loss = eval_temp(temp, oof_logits, oof_labels)
    if loss < best_loss:
        best_loss = loss
        best_temp = temp

print(f"Best temperature: {best_temp:.4f}")
calibrated_probs = softmax(oof_logits, best_temp)
cv_logloss = log_loss(oof_labels, calibrated_probs)
print(f"Cross-validated log loss after calibration: {cv_logloss:.6f}")

# -------------------- Test predictions --------------------
test_probs = softmax(test_logits, best_temp)

# -------------------- Submission --------------------
submission_df = pd.DataFrame({
    'id': test_df['id'],
    'winner_model_a': test_probs[:, 0],
    'winner_model_b': test_probs[:, 1],
    'winner_tie': test_probs[:, 2]
})

os.makedirs('./submission', exist_ok=True)
submission_df.to_csv('./submission/submission.csv', index=False)
print("Submission saved to ./submission/submission.csv")