import os
import re
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.svm import LinearSVR
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score
import lightgbm as lgb

# Fixed randomness
SEED = 42
np.random.seed(SEED)

# Directories
INPUT_DIR = "input"
OUTPUT_DIR = "submission"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- Load data ----------
train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv"))

train['full_text'] = train['full_text'].fillna('')
test['full_text'] = test['full_text'].fillna('')

y = train['score'].values
train_ids = train['essay_id'].values
test_ids = test['essay_id'].values

# ---------- Linguistic feature extraction ----------
def extract_linguistic_features(texts):
    features = []
    for txt in texts:
        txt = str(txt)

        # Basic counts
        words = txt.split()
        word_count = len(words)
        char_count = len(txt)

        # Sentence count
        sentences = [s for s in re.split(r'[.!?]+', txt) if s.strip()]
        sentence_count = len(sentences)

        # Paragraph count
        paragraphs = [p for p in txt.split('\n') if p.strip()]
        paragraph_count = len(paragraphs) if paragraphs else 1

        # Punctuation counts
        comma_count = txt.count(',')
        question_count = txt.count('?')
        exclam_count = txt.count('!')

        # Syllable count (rough regex based)
        lower_txt = txt.lower()
        syllable_count = len(re.findall(r'[aeiouy]+', lower_txt))

        # Word length
        avg_word_len = char_count / max(word_count, 1)

        # Unique words & TTR
        unique_words = set(words)
        unique_word_count = len(unique_words)
        ttr = unique_word_count / max(word_count, 1)

        # Readability formulas (use safe denominators)
        w_safe = max(word_count, 1)
        s_safe = max(sentence_count, 1)
        flesch_reading_ease = 206.835 - 1.015 * (w_safe / s_safe) - 84.6 * (syllable_count / w_safe)
        flesch_kincaid_grade = 0.39 * (w_safe / s_safe) + 11.8 * (syllable_count / w_safe) - 15.59
        automated_readability_index = 4.71 * (char_count / w_safe) + 0.5 * (w_safe / s_safe) - 21.43

        features.append([
            word_count, char_count, sentence_count, paragraph_count,
            comma_count, question_count, exclam_count,
            syllable_count, avg_word_len, unique_word_count, ttr,
            flesch_reading_ease, flesch_kincaid_grade, automated_readability_index
        ])
    return np.array(features)

print("Extracting linguistic features...")
ling_train = extract_linguistic_features(train['full_text'].values)
ling_test  = extract_linguistic_features(test['full_text'].values)

# ---------- Global SVD features (word TF‑IDF) ----------
print("Creating global SVD features...")
all_texts = pd.concat([train['full_text'], test['full_text']], ignore_index=True)

word_vect_global = TfidfVectorizer(
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    ngram_range=(1, 2),
    min_df=10,
    max_df=0.95,
    sublinear_tf=True,
    stop_words='english'
)
X_word_all = word_vect_global.fit_transform(all_texts)
svd = TruncatedSVD(n_components=50, random_state=SEED)
X_svd_all = svd.fit_transform(X_word_all)

X_svd_train = X_svd_all[:len(train)]
X_svd_test  = X_svd_all[len(train):]

# ---------- Cross‑validation splits ----------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
folds = list(skf.split(train, y))

# ---------- Base models: Ridge & LinearSVR ----------
ridge_oof = np.zeros(len(train))
svr_oof   = np.zeros(len(train))
ridge_test = np.zeros(len(test))
svr_test   = np.zeros(len(test))

for fold, (trn_idx, val_idx) in enumerate(folds):
    print(f"\nFold {fold+1}/5")

    X_text_trn = train['full_text'].iloc[trn_idx]
    X_text_val = train['full_text'].iloc[val_idx]
    y_trn = y[trn_idx]
    y_val = y[val_idx]

    # Word TF‑IDF (fold‑specific)
    word_vect = TfidfVectorizer(
        strip_accents='unicode',
        analyzer='word',
        token_pattern=r'\w{1,}',
        ngram_range=(1, 2),
        min_df=10,
        max_df=0.95,
        sublinear_tf=True,
        stop_words='english'
    )
    X_word_trn = word_vect.fit_transform(X_text_trn)
    X_word_val = word_vect.transform(X_text_val)
    X_word_ts  = word_vect.transform(test['full_text'])

    # Character TF‑IDF (fold‑specific)
    char_vect = TfidfVectorizer(
        strip_accents='unicode',
        analyzer='char',
        ngram_range=(3, 5),
        min_df=5,
        max_df=0.95,
        sublinear_tf=True
    )
    X_char_trn = char_vect.fit_transform(X_text_trn)
    X_char_val = char_vect.transform(X_text_val)
    X_char_ts  = char_vect.transform(test['full_text'])

    # Scale linguistic features
    scaler = StandardScaler()
    ling_trn_scaled = scaler.fit_transform(ling_train[trn_idx])
    ling_val_scaled = scaler.transform(ling_train[val_idx])
    ling_ts_scaled  = scaler.transform(ling_test)

    # Combine all features (sparse + dense)
    X_trn = sparse.hstack([X_word_trn, X_char_trn, ling_trn_scaled])
    X_val = sparse.hstack([X_word_val, X_char_val, ling_val_scaled])
    X_ts  = sparse.hstack([X_word_ts,  X_char_ts,  ling_ts_scaled])

    # Ridge regression
    ridge = Ridge(alpha=1.0, random_state=SEED)
    ridge.fit(X_trn, y_trn)
    ridge_oof[val_idx] = ridge.predict(X_val)
    ridge_test += ridge.predict(X_ts) / len(folds)

    # Linear SVR
    svr = LinearSVR(epsilon=0.1, C=1.0, max_iter=5000, dual=True, random_state=SEED)
    svr.fit(X_trn, y_trn)
    svr_oof[val_idx] = svr.predict(X_val)
    svr_test += svr.predict(X_ts) / len(folds)

# Base model OOF QWK
print("\nBase models OOF QWK (rounded):")
print("Ridge:    {:.6f}".format(cohen_kappa_score(y, ridge_oof.round(), weights='quadratic')))
print("LinearSVR:{:.6f}".format(cohen_kappa_score(y, svr_oof.round(), weights='quadratic')))

# ---------- Stacking with LightGBM ----------
print("\nStacking with LightGBM...")

# Build stacking features: linguistic (raw) + base OOF + SVD
X_stack_train = np.hstack([
    ling_train,
    ridge_oof.reshape(-1, 1),
    svr_oof.reshape(-1, 1),
    X_svd_train
])
X_stack_test = np.hstack([
    ling_test,
    ridge_test.reshape(-1, 1),
    svr_test.reshape(-1, 1),
    X_svd_test
])

lgb_oof = np.zeros(len(train))
lgb_test = np.zeros(len(test))

for fold, (trn_idx, val_idx) in enumerate(folds):
    print(f"LightGBM fold {fold+1}/5")
    X_trn = X_stack_train[trn_idx]
    X_val = X_stack_train[val_idx]
    y_trn = y[trn_idx]
    y_val = y[val_idx]

    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        objective='rmse',
        random_state=SEED,
        n_jobs=-1,
        verbose=-1
    )
    model.fit(
        X_trn, y_trn,
        eval_set=[(X_val, y_val)],
        eval_metric='rmse',
        callbacks=[lgb.early_stopping(stopping_rounds=50)]
    )
    lgb_oof[val_idx] = model.predict(X_val, num_iteration=model.best_iteration_)
    lgb_test += model.predict(X_stack_test, num_iteration=model.best_iteration_) / len(folds)

print("LightGBM OOF QWK (raw, rounded): {:.6f}".format(
    cohen_kappa_score(y, lgb_oof.round(), weights='quadratic')))

# ---------- Threshold optimization ----------
def kappa_obj(thresholds, preds, true):
    # Ensure thresholds are sorted for consistent digitizing
    thresholds = np.sort(thresholds)
    disc = np.digitize(preds, thresholds) + 1
    return -cohen_kappa_score(true, disc, weights='quadratic')

init_thresholds = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
res = minimize(
    kappa_obj, init_thresholds, args=(lgb_oof, y),
    method='Nelder-Mead', options={'maxiter': 1000}
)
opt_thresholds = np.sort(res.x)
print("\nOptimized thresholds:", opt_thresholds)

lgb_oof_discrete = np.digitize(lgb_oof, opt_thresholds) + 1
final_kappa = cohen_kappa_score(y, lgb_oof_discrete, weights='quadratic')
print(f"Optimized OOF QWK: {final_kappa:.6f}")

# ---------- Final test predictions ----------
test_preds = np.digitize(lgb_test, opt_thresholds) + 1

# ---------- Write submission ----------
sub = pd.DataFrame({'essay_id': test_ids, 'score': test_preds})
sub.to_csv(os.path.join(OUTPUT_DIR, "submission.csv"), index=False)
print("Submission saved to submission/submission.csv")