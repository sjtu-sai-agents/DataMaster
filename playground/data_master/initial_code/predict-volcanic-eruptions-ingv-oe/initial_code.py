#!/usr/bin/env python3
import os
import sys
import numpy as np
import pandas as pd
import scipy.stats as sp_stats
from scipy import stats
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor
from joblib import Parallel, delayed
import lightgbm as lgb
try:
    from catboost import CatBoostRegressor
    USE_CATBOOST = True
except ImportError:
    USE_CATBOOST = False
import warnings
warnings.filterwarnings('ignore')

# ----------------------------------------------------------------------
# Constants
SEED = 42
N_FOLDS = 5
DOWN_FACTOR = 10                # downsampling factor for FFT and correlation
TAIL_CORR_SIZE = 10000          # number of points for tail correlation

# Paths
INPUT_DIR = "./input"
TRAIN_CSV = os.path.join(INPUT_DIR, "train.csv")
SAMPLE_SUB = os.path.join(INPUT_DIR, "sample_submission.csv")
TRAIN_DATA_DIR = os.path.join(INPUT_DIR, "train")
TEST_DATA_DIR = os.path.join(INPUT_DIR, "test")
SUBMISSION_DIR = "./submission"
SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# ----------------------------------------------------------------------
# Feature extraction helpers (all operate on 1D arrays)
def compute_basic_stats(x, prefix):
    """Return dict with mean, std, min, max, median, var, percentiles,
       skew, kurt, mad, zero_crossing, energy."""
    x = np.asarray(x, dtype=np.float32)
    stats = {}
    stats[f"{prefix}_mean"]   = np.mean(x)
    stats[f"{prefix}_std"]    = np.std(x)
    stats[f"{prefix}_min"]    = np.min(x)
    stats[f"{prefix}_max"]    = np.max(x)
    stats[f"{prefix}_median"] = np.median(x)
    stats[f"{prefix}_var"]    = np.var(x)

    percentiles = [1, 5, 25, 50, 75, 95, 99]
    p_vals = np.percentile(x, percentiles)
    for p, v in zip(percentiles, p_vals):
        stats[f"{prefix}_p{p}"] = v

    if len(x) > 0:
        stats[f"{prefix}_skew"] = sp_stats.skew(x)
        stats[f"{prefix}_kurt"] = sp_stats.kurtosis(x)
    else:
        stats[f"{prefix}_skew"] = 0.0
        stats[f"{prefix}_kurt"] = 0.0

    med = np.median(x)
    stats[f"{prefix}_mad"] = np.median(np.abs(x - med))

    if len(x) > 1:
        zc = ((x[:-1] * x[1:]) < 0).sum() / (len(x)-1)
    else:
        zc = 0.0
    stats[f"{prefix}_zero_crossing"] = zc

    stats[f"{prefix}_energy"] = np.mean(x**2)
    return stats

def compute_diff_stats(diff, prefix):
    diff = np.asarray(diff, dtype=np.float32)
    stats = {}
    stats[f"{prefix}_mean"]   = np.mean(diff)
    stats[f"{prefix}_std"]    = np.std(diff)
    stats[f"{prefix}_min"]    = np.min(diff)
    stats[f"{prefix}_max"]    = np.max(diff)
    stats[f"{prefix}_median"] = np.median(diff)
    return stats

def compute_autocorr(x, lags, prefix):
    x = np.asarray(x, dtype=np.float32)
    stats = {}
    for lag in lags:
        if len(x) > lag:
            a = x[lag:]
            b = x[:-lag]
            if np.std(a) > 1e-12 and np.std(b) > 1e-12:
                corr = np.corrcoef(a, b)[0, 1]
            else:
                corr = 0.0
        else:
            corr = 0.0
        stats[f"{prefix}_{lag}"] = corr
    return stats

def compute_fft_features(x, fs=100.0, down_factor=10, prefix=""):
    x = np.asarray(x, dtype=np.float32)
    # downsample
    if len(x) >= down_factor:
        y = x[::down_factor]
    else:
        y = x
    y = y - np.mean(y)
    n = len(y)
    stats = {}
    if n < 2:
        stats.update({f"{prefix}_dom_freq": 0.0, f"{prefix}_dom_mag": 0.0,
                      f"{prefix}_centroid": 0.0, f"{prefix}_spread": 0.0,
                      f"{prefix}_flatness": 0.0})
        return stats

    fft_vals = np.fft.rfft(y)
    mag = np.abs(fft_vals)
    freq = np.fft.rfftfreq(n, d=1.0/(fs/down_factor))

    # dominant frequency & magnitude
    idx = np.argmax(mag)
    stats[f"{prefix}_dom_freq"] = freq[idx]
    stats[f"{prefix}_dom_mag"]  = mag[idx]

    # spectral centroid
    sum_mag = np.sum(mag)
    if sum_mag > 0:
        centroid = np.sum(freq * mag) / sum_mag
    else:
        centroid = 0.0
    stats[f"{prefix}_centroid"] = centroid

    # spectral spread
    if sum_mag > 0:
        spread = np.sqrt(np.sum((freq - centroid)**2 * mag) / sum_mag)
    else:
        spread = 0.0
    stats[f"{prefix}_spread"] = spread

    # spectral flatness
    mag_nonzero = mag[mag > 0]
    if len(mag_nonzero) > 0:
        log_mag = np.log(mag_nonzero)
        geo_mean = np.exp(np.mean(log_mag))
        arith_mean = np.mean(mag_nonzero)
        flatness = geo_mean / (arith_mean + 1e-12)
    else:
        flatness = 0.0
    stats[f"{prefix}_flatness"] = flatness
    return stats

def compute_block_stats(x, n_blocks, prefix):
    x = np.asarray(x, dtype=np.float32)
    blocks = np.array_split(x, n_blocks)
    stats = {}
    for i, blk in enumerate(blocks):
        stats[f"{prefix}_block{i+1}_mean"] = np.mean(blk)
        stats[f"{prefix}_block{i+1}_std"]  = np.std(blk)
    return stats

def compute_correlation_stats(data, prefix):
    """data: (n_samples, 10). Return 8 statistics on correlations."""
    if data.shape[0] < 2:
        return {f"{prefix}_corr_mean": 0.0, f"{prefix}_corr_std": 0.0,
                f"{prefix}_corr_min": 0.0, f"{prefix}_corr_max": 0.0,
                f"{prefix}_corr_abs_mean": 0.0, f"{prefix}_corr_abs_std": 0.0,
                f"{prefix}_corr_abs_min": 0.0, f"{prefix}_corr_abs_max": 0.0}
    corr = np.corrcoef(data, rowvar=False)   # 10x10
    triu_idx = np.triu_indices(10, k=1)
    corr_vals = corr[triu_idx]
    abs_corr_vals = np.abs(corr_vals)
    stats = {}
    stats[f"{prefix}_corr_mean"] = np.mean(corr_vals)
    stats[f"{prefix}_corr_std"]  = np.std(corr_vals)
    stats[f"{prefix}_corr_min"]  = np.min(corr_vals)
    stats[f"{prefix}_corr_max"]  = np.max(corr_vals)
    stats[f"{prefix}_corr_abs_mean"] = np.mean(abs_corr_vals)
    stats[f"{prefix}_corr_abs_std"]  = np.std(abs_corr_vals)
    stats[f"{prefix}_corr_abs_min"]  = np.min(abs_corr_vals)
    stats[f"{prefix}_corr_abs_max"]  = np.max(abs_corr_vals)
    return stats

# ----------------------------------------------------------------------
# Main feature extraction for one segment file
def extract_features(file_path):
    try:
        df = pd.read_csv(file_path, dtype=np.float32)
        data = df.values                     # (n_samples, 10)
        data = np.nan_to_num(data, copy=False)
        n = data.shape[0]
        if n == 0:
            return None
        features = {}

        # ---------- Per-sensor features ----------
        for s in range(10):
            prefix = f"s{s+1}"
            x = data[:, s]

            # Whole signal basic stats
            features.update(compute_basic_stats(x, prefix))

            # First differences
            diff = np.diff(x)
            features.update(compute_diff_stats(diff, f"{prefix}_diff"))

            # Autocorrelation
            lags = [1, 5, 50, 500]
            features.update(compute_autocorr(x, lags, f"{prefix}_autocorr"))

            # FFT (downsampling inside)
            features.update(compute_fft_features(x, fs=100.0, down_factor=DOWN_FACTOR,
                                                 prefix=f"{prefix}_fft"))

            # Windowed basic stats
            # tail windows
            tail_sizes = [500, 2000, 10000, 30000]
            for ws in tail_sizes:
                window = x[-ws:] if n >= ws else x
                features.update(compute_basic_stats(window, f"{prefix}_tail{ws}"))

            # head windows
            head_sizes = [2000, 10000]
            for ws in head_sizes:
                window = x[:ws] if n >= ws else x
                features.update(compute_basic_stats(window, f"{prefix}_head{ws}"))

            # middle windows
            mid_sizes = [2000, 10000]
            for ws in mid_sizes:
                if n >= ws:
                    mid_start = (n - ws) // 2
                    window = x[mid_start:mid_start+ws]
                else:
                    window = x
                features.update(compute_basic_stats(window, f"{prefix}_mid{ws}"))

            # Block stats (5 blocks)
            features.update(compute_block_stats(x, 5, prefix))

        # ---------- Cross-sensor correlation features ----------
        # full downsampled
        full_down = data[::DOWN_FACTOR, :]
        features.update(compute_correlation_stats(full_down, "corr_full"))
        # tail downsampled
        if n >= TAIL_CORR_SIZE:
            tail_part = data[-TAIL_CORR_SIZE:, :]
        else:
            tail_part = data
        tail_down = tail_part[::DOWN_FACTOR, :]
        features.update(compute_correlation_stats(tail_down, "corr_tail"))

        # ---------- Row-mean features ----------
        row_mean = np.mean(data, axis=1)
        # basic stats on whole row-mean
        features.update(compute_basic_stats(row_mean, "row_mean"))
        # tail windows for row-mean
        for ws in [2000, 10000, 30000]:
            window = row_mean[-ws:] if len(row_mean) >= ws else row_mean
            features.update(compute_basic_stats(window, f"row_mean_tail{ws}"))
        # block stats for row-mean
        features.update(compute_block_stats(row_mean, 5, "row_mean"))

        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}", file=sys.stderr)
        return None

def process_segment(seg_id, data_dir):
    file_path = os.path.join(data_dir, f"{seg_id}.csv")
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found", file=sys.stderr)
        return None
    feats = extract_features(file_path)
    if feats is None:
        return None
    feats['segment_id'] = seg_id
    return feats

# ----------------------------------------------------------------------
# Load metadata
train_meta = pd.read_csv(TRAIN_CSV)
train_ids = train_meta['segment_id'].values
y_train_all = train_meta['time_to_eruption'].values.astype(np.float32)

sub_sample = pd.read_csv(SAMPLE_SUB)
test_ids = sub_sample['segment_id'].values

# ----------------------------------------------------------------------
# Extract features for training set
print("Processing training segments...")
train_results = Parallel(n_jobs=-1)(
    delayed(process_segment)(seg_id, TRAIN_DATA_DIR) for seg_id in train_ids
)
train_feat_dict = {res['segment_id']: res for res in train_results if res is not None}
# keep only successful ids, in original order
train_ids_success = [sid for sid in train_ids if sid in train_feat_dict]
train_rows = [train_feat_dict[sid] for sid in train_ids_success]
train_feat_df = pd.DataFrame(train_rows)
train_feat_df = train_feat_df.set_index('segment_id').loc[train_ids_success].reset_index()
y_train = y_train_all[np.isin(train_ids, train_ids_success)]

# Fill NaNs and define feature columns
train_feat_df.fillna(0.0, inplace=True)
FEATURE_COLS = [c for c in train_feat_df.columns if c != 'segment_id']
X_train = train_feat_df[FEATURE_COLS].astype(np.float32).values
y_train = y_train.astype(np.float32)

print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")

# ----------------------------------------------------------------------
# Extract features for test set
print("\nProcessing test segments...")
test_results = Parallel(n_jobs=-1)(
    delayed(process_segment)(seg_id, TEST_DATA_DIR) for seg_id in test_ids
)
test_feat_dict = {res['segment_id']: res for res in test_results if res is not None}
test_rows = []
for sid in test_ids:
    if sid in test_feat_dict:
        d = test_feat_dict[sid].copy()
    else:
        d = {'segment_id': sid}
    # ensure all feature columns exist
    for col in FEATURE_COLS:
        if col not in d:
            d[col] = 0.0
    test_rows.append(d)
test_feat_df = pd.DataFrame(test_rows)
test_feat_df.fillna(0.0, inplace=True)
# enforce column order
test_feat_df = test_feat_df[['segment_id'] + FEATURE_COLS]
X_test = test_feat_df[FEATURE_COLS].astype(np.float32).values

print(f"Test set: {X_test.shape[0]} samples")

# ----------------------------------------------------------------------
# Model parameters
lgb_params = {
    'objective': 'regression_l1',
    'metric': 'mae',
    'boosting_type': 'gbdt',
    'learning_rate': 0.03,
    'num_leaves': 63,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.0,
    'reg_lambda': 0.0,
    'random_state': SEED,
    'n_jobs': -1,
    'verbose': -1
}

et_params = {
    'n_estimators': 600,
    'max_features': 0.6,
    'bootstrap': False,
    'random_state': 1337,
    'n_jobs': -1,
    'verbose': 0
}

if USE_CATBOOST:
    cb_params = {
        'iterations': 6000,
        'depth': 8,
        'learning_rate': 0.05,
        'l2_leaf_reg': 3.0,
        'bootstrap_type': 'Bernoulli',
        'subsample': 0.8,
        'loss_function': 'MAE',
        'eval_metric': 'MAE',
        'random_seed': 2025,
        'verbose': False,
        'allow_writing_files': False,
        'early_stopping_rounds': 300
    }

# ----------------------------------------------------------------------
# Cross‑validation setup
try:
    bins = np.quantile(y_train, np.linspace(0, 1, 11))
    bins[-1] += 1e-6
    y_bin = np.digitize(y_train, bins)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = skf.split(X_train, y_bin)
except Exception:
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = kf.split(X_train)

# OOF storage
oof_lgb = np.zeros(len(y_train), dtype=np.float32)
oof_et  = np.zeros(len(y_train), dtype=np.float32)
if USE_CATBOOST:
    oof_cb = np.zeros(len(y_train), dtype=np.float32)

# Test predictions accumulation
test_preds_lgb = np.zeros(X_test.shape[0], dtype=np.float32)
test_preds_et  = np.zeros(X_test.shape[0], dtype=np.float32)
if USE_CATBOOST:
    test_preds_cb = np.zeros(X_test.shape[0], dtype=np.float32)

# ----------------------------------------------------------------------
# Cross‑validation loop
for fold, (trn_idx, val_idx) in enumerate(folds):
    print(f"\n{'='*40} Fold {fold+1} / {N_FOLDS} {'='*40}")
    X_tr, X_val = X_train[trn_idx], X_train[val_idx]
    y_tr, y_val = y_train[trn_idx], y_train[val_idx]

    # ----- LightGBM -----
    print("Training LightGBM ...")
    train_data = lgb.Dataset(X_tr, label=y_tr)
    val_data   = lgb.Dataset(X_val, label=y_val, reference=train_data)
    lgb_model = lgb.train(
        params=lgb_params,
        train_set=train_data,
        num_boost_round=5000,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        callbacks=[lgb.early_stopping(stopping_rounds=200), lgb.log_evaluation(200)]
    )
    oof_lgb[val_idx] = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)
    test_preds_lgb += lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration) / N_FOLDS

    # ----- ExtraTrees -----
    print("Training ExtraTrees ...")
    et_model = ExtraTreesRegressor(**et_params)
    et_model.fit(X_tr, y_tr)
    oof_et[val_idx] = et_model.predict(X_val)
    test_preds_et += et_model.predict(X_test) / N_FOLDS

    # ----- CatBoost -----
    if USE_CATBOOST:
        print("Training CatBoost ...")
        cb_model = CatBoostRegressor(**cb_params)
        cb_model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=100)
        oof_cb[val_idx] = cb_model.predict(X_val)
        test_preds_cb += cb_model.predict(X_test) / N_FOLDS

    # Fold MAE
    mae_lgb_f = mean_absolute_error(y_val, oof_lgb[val_idx])
    mae_et_f  = mean_absolute_error(y_val, oof_et[val_idx])
    print(f"Fold {fold+1} | LGB MAE: {mae_lgb_f:.4f}, ET MAE: {mae_et_f:.4f}", end="")
    if USE_CATBOOST:
        mae_cb_f = mean_absolute_error(y_val, oof_cb[val_idx])
        print(f", CB MAE: {mae_cb_f:.4f}")
    else:
        print()

# ----------------------------------------------------------------------
# OOF scores
mae_lgb = mean_absolute_error(y_train, oof_lgb)
mae_et  = mean_absolute_error(y_train, oof_et)
print("\n" + "="*50)
print(f"OOF MAE - LightGBM : {mae_lgb:.4f}")
print(f"OOF MAE - ExtraTrees: {mae_et:.4f}")
if USE_CATBOOST:
    mae_cb = mean_absolute_error(y_train, oof_cb)
    print(f"OOF MAE - CatBoost : {mae_cb:.4f}")

# ----------------------------------------------------------------------
# Stacking with Ridge
base_oof = [oof_lgb, oof_et]
if USE_CATBOOST:
    base_oof.append(oof_cb)
base_oof = np.column_stack(base_oof)

stacker = Ridge(alpha=1.0, random_state=SEED)
stacker.fit(base_oof, y_train)
stacked_oof = stacker.predict(base_oof)
mae_stacked = mean_absolute_error(y_train, stacked_oof)
print(f"\nStacked OOF MAE: {mae_stacked:.4f}")
print("Stacker coefficients:", stacker.coef_, "intercept:", stacker.intercept_)

# ----------------------------------------------------------------------
# Test predictions from stacking
base_test = [test_preds_lgb, test_preds_et]
if USE_CATBOOST:
    base_test.append(test_preds_cb)
base_test = np.column_stack(base_test)
test_pred = stacker.predict(base_test)
test_pred = np.maximum(test_pred, 0.0)   # time cannot be negative

# ----------------------------------------------------------------------
# Save submission
submission_df = pd.DataFrame({
    'segment_id': test_ids,
    'time_to_eruption': test_pred
})
submission_df.to_csv(SUBMISSION_FILE, index=False)
print(f"\nSubmission saved to {SUBMISSION_FILE}")

print("\nDone.")