import os
import random
import warnings
from tqdm import tqdm

import numpy as np
import pandas as pd

from PIL import Image, ImageFile, ImageOps

import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

import timm

import lightgbm as lgb
import catboost as cb
import xgboost as xgb

from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# ----------------------------------------------------------------------
# Reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] = str(seed)

# ----------------------------------------------------------------------
# Paths
INPUT_DIR = "./input"
TRAIN_IMG_DIR = os.path.join(INPUT_DIR, "train")
TEST_IMG_DIR = os.path.join(INPUT_DIR, "test")
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings('ignore')

# ----------------------------------------------------------------------
# Hand‑crafted image statistics
def compute_img_stats(image_folder, ids, output_file):
    """Compute 17 statistical features from each image and cache."""
    if os.path.exists(output_file):
        print(f"Loading cached image stats from {output_file}")
        return pd.read_csv(output_file)

    print(f"Computing image stats for {len(ids)} images...")
    stats = []
    for img_id in tqdm(ids, total=len(ids)):
        img_path = os.path.join(image_folder, f"{img_id}.jpg")
        with Image.open(img_path) as img:
            img = ImageOps.exif_transpose(img)
            img = img.convert('RGB')

            width, height = img.size
            aspect = width / height if height > 0 else 0
            area = width * height

            img_array = np.array(img, dtype=np.float32) / 255.0

            # Grayscale intensity
            gray = 0.2989 * img_array[:,:,0] + 0.5870 * img_array[:,:,1] + 0.1140 * img_array[:,:,2]
            intensity_mean = gray.mean()
            intensity_std = gray.std()
            intensity_min = gray.min()
            intensity_max = gray.max()
            brightness_range = intensity_max - intensity_min

            # RGB channel stats
            r_mean = img_array[:,:,0].mean()
            g_mean = img_array[:,:,1].mean()
            b_mean = img_array[:,:,2].mean()
            r_std = img_array[:,:,0].std()
            g_std = img_array[:,:,1].std()
            b_std = img_array[:,:,2].std()

            # Saturation (mean of per‑pixel std across channels)
            saturation = img_array.std(axis=2).mean()

            stats.append({
                'Id': img_id,
                'img_width': width,
                'img_height': height,
                'img_aspect_ratio': aspect,
                'img_area': area,
                'img_log_area': np.log1p(area),
                'img_mean_intensity': intensity_mean,
                'img_std_intensity': intensity_std,
                'img_min_intensity': intensity_min,
                'img_max_intensity': intensity_max,
                'img_brightness_range': brightness_range,
                'img_color_mean_r': r_mean,
                'img_color_mean_g': g_mean,
                'img_color_mean_b': b_mean,
                'img_color_std_r': r_std,
                'img_color_std_g': g_std,
                'img_color_std_b': b_std,
                'img_mean_saturation': saturation
            })

    df = pd.DataFrame(stats)
    df.to_csv(output_file, index=False)
    return df

# ----------------------------------------------------------------------
# CNN embedding extraction
class ImageDataset(data.Dataset):
    def __init__(self, image_folder, ids, transform):
        self.image_folder = image_folder
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_path = os.path.join(self.image_folder, f"{img_id}.jpg")
        with Image.open(img_path) as img:
            img = ImageOps.exif_transpose(img)
            img = img.convert('RGB')
            if self.transform:
                img = self.transform(img)
        return img, img_id

def extract_embeddings(model_name, image_folder, ids, output_file, device,
                       batch_size=32, num_workers=8):
    """Extract embeddings from a pretrained timm model and cache."""
    if os.path.exists(output_file):
        print(f"Loading cached embeddings from {output_file}")
        return pd.read_csv(output_file)

    print(f"Extracting embeddings with {model_name} for {len(ids)} images...")

    # Input size
    if 'efficientnet' in model_name.lower():
        image_size = 380
    else:
        image_size = 224   # convnext_base

    transform = transforms.Compose([
        transforms.Resize(int(image_size * 1.1), interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    dataset = ImageDataset(image_folder, ids, transform=transform)
    dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers, pin_memory=True)

    model = timm.create_model(model_name, pretrained=True,
                              num_classes=0, global_pool='avg')
    model.to(device)
    model.eval()

    all_embeddings = []
    all_ids = []
    with torch.no_grad():
        for imgs, ids_batch in tqdm(dataloader, total=len(dataloader)):
            imgs = imgs.to(device, non_blocking=True)
            emb = model(imgs).cpu().numpy()
            all_embeddings.append(emb)
            all_ids.extend(ids_batch)

    embeddings = np.vstack(all_embeddings)
    df = pd.DataFrame(embeddings)
    df.columns = [f'{model_name}_emb_{i}' for i in range(embeddings.shape[1])]
    df.insert(0, 'Id', all_ids)
    df.to_csv(output_file, index=False)

    del model
    torch.cuda.empty_cache()
    return df

# ----------------------------------------------------------------------
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load CSVs and normalise column names
    train_df = pd.read_csv(os.path.join(INPUT_DIR, 'train.csv'))
    test_df = pd.read_csv(os.path.join(INPUT_DIR, 'test.csv'))
    train_df.columns = [c.replace(' ', '_') for c in train_df.columns]
    test_df.columns = [c.replace(' ', '_') for c in test_df.columns]

    train_ids = train_df['Id'].tolist()
    test_ids = test_df['Id'].tolist()

    # 1. Hand‑crafted image features
    train_img_stats_file = os.path.join(WORKING_DIR, 'train_img_stats.csv')
    test_img_stats_file = os.path.join(WORKING_DIR, 'test_img_stats.csv')
    train_img_stats = compute_img_stats(TRAIN_IMG_DIR, train_ids, train_img_stats_file)
    test_img_stats = compute_img_stats(TEST_IMG_DIR, test_ids, test_img_stats_file)

    # 2. CNN embeddings (two models)
    model_names = ['tf_efficientnet_b4_ns', 'convnext_base']
    for model_name in model_names:
        train_emb_file = os.path.join(WORKING_DIR, f'train_{model_name}_embeddings.csv')
        test_emb_file = os.path.join(WORKING_DIR, f'test_{model_name}_embeddings.csv')
        if not os.path.exists(train_emb_file):
            extract_embeddings(model_name, TRAIN_IMG_DIR, train_ids,
                               train_emb_file, device)
        if not os.path.exists(test_emb_file):
            extract_embeddings(model_name, TEST_IMG_DIR, test_ids,
                               test_emb_file, device)

    # 3. Merge all features
    X_train = train_df.drop(['Pawpularity'], axis=1).copy()
    X_test = test_df.copy()
    y_train = train_df['Pawpularity'].astype(np.float32).copy()

    X_train = X_train.merge(train_img_stats, on='Id', how='left', validate='one_to_one')
    X_test = X_test.merge(test_img_stats, on='Id', how='left', validate='one_to_one')

    for model_name in model_names:
        train_emb = pd.read_csv(os.path.join(WORKING_DIR, f'train_{model_name}_embeddings.csv'))
        test_emb = pd.read_csv(os.path.join(WORKING_DIR, f'test_{model_name}_embeddings.csv'))
        X_train = X_train.merge(train_emb, on='Id', how='left', validate='one_to_one')
        X_test = X_test.merge(test_emb, on='Id', how='left', validate='one_to_one')

    # Keep Id for submission
    train_ids_save = X_train['Id'].copy()
    test_ids_save = X_test['Id'].copy()
    X_train = X_train.drop('Id', axis=1)
    X_test = X_test.drop('Id', axis=1)

    # Ensure numeric and fill NaNs
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    feat_means = X_train.mean()
    X_train = X_train.fillna(feat_means)
    X_test = X_test.fillna(feat_means)

    X = X_train.values
    y = y_train.values
    X_test_vals = X_test.values

    # ------------------------------------------------------------------
    # 5‑fold CV with three gradient boosting models
    n_folds = 5
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    oof_lgb = np.zeros(len(X), dtype=np.float32)
    oof_cat = np.zeros(len(X), dtype=np.float32)
    oof_xgb = np.zeros(len(X), dtype=np.float32)

    test_lgb = np.zeros(len(X_test_vals), dtype=np.float32)
    test_cat = np.zeros(len(X_test_vals), dtype=np.float32)
    test_xgb = np.zeros(len(X_test_vals), dtype=np.float32)

    # Hyperparameters (taken from gold solution)
    lgb_params = {
        'n_estimators': 5000,
        'learning_rate': 0.01,
        'num_leaves': 64,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'min_child_samples': 20,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'random_state': seed,
        'n_jobs': -1,
        'metric': 'rmse',
        'verbose': -1
    }
    cat_params = {
        'depth': 8,
        'learning_rate': 0.03,
        'loss_function': 'RMSE',
        'iterations': 5000,
        'random_seed': seed,
        'od_type': 'Iter',
        'od_wait': 200,
        'l2_leaf_reg': 3.0,
        'verbose': 0
    }
    xgb_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'learning_rate': 0.02,
        'max_depth': 8,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 5,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'gamma': 0.0,
        'tree_method': 'hist',
        'random_state': seed
    }

    for fold, (trn_idx, val_idx) in enumerate(kf.split(X, y)):
        print(f"\nFold {fold+1}/{n_folds}")
        X_tr, X_va = X[trn_idx], X[val_idx]
        y_tr, y_va = y[trn_idx], y[val_idx]

        # ---- LightGBM ----
        lgb_train = lgb.Dataset(X_tr, y_tr)
        lgb_valid = lgb.Dataset(X_va, y_va, reference=lgb_train)
        model_lgb = lgb.train(
            lgb_params,
            lgb_train,
            valid_sets=[lgb_valid],
            callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False)]
        )
        oof_lgb[val_idx] = model_lgb.predict(X_va, num_iteration=model_lgb.best_iteration)
        test_lgb += model_lgb.predict(X_test_vals, num_iteration=model_lgb.best_iteration) / n_folds
        print(f" LGB fold RMSE: {np.sqrt(mean_squared_error(y_va, oof_lgb[val_idx])):.4f}")

        # ---- CatBoost ----
        model_cat = cb.CatBoostRegressor(**cat_params)
        model_cat.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=200, verbose=0)
        oof_cat[val_idx] = model_cat.predict(X_va)
        test_cat += model_cat.predict(X_test_vals) / n_folds
        print(f" CatBoost fold RMSE: {np.sqrt(mean_squared_error(y_va, oof_cat[val_idx])):.4f}")

        # ---- XGBoost ----
        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dvalid = xgb.DMatrix(X_va, label=y_va)
        dtest = xgb.DMatrix(X_test_vals)
        model_xgb = xgb.train(
            xgb_params,
            dtrain,
            evals=[(dtrain, 'train'), (dvalid, 'valid')],
            num_boost_round=5000,
            early_stopping_rounds=200,
            verbose_eval=False
        )
        oof_xgb[val_idx] = model_xgb.predict(dvalid)
        test_xgb += model_xgb.predict(dtest) / n_folds
        print(f" XGB fold RMSE: {np.sqrt(mean_squared_error(y_va, oof_xgb[val_idx])):.4f}")

    # Overall OOF RMSE
    lgb_rmse = np.sqrt(mean_squared_error(y, oof_lgb))
    cat_rmse = np.sqrt(mean_squared_error(y, oof_cat))
    xgb_rmse = np.sqrt(mean_squared_error(y, oof_xgb))
    print("\nOverall OOF RMSE:")
    print(f" LightGBM: {lgb_rmse:.4f}")
    print(f" CatBoost: {cat_rmse:.4f}")
    print(f" XGBoost:  {xgb_rmse:.4f}")

    # ------------------------------------------------------------------
    # Stacking with Ridge
    stack_X = np.column_stack((oof_lgb, oof_cat, oof_xgb))
    stack_model = Ridge(alpha=1.0, random_state=seed)
    stack_model.fit(stack_X, y)
    stack_rmse = np.sqrt(mean_squared_error(y, stack_model.predict(stack_X)))
    print(f"\nStacking Ridge RMSE on OOF: {stack_rmse:.4f}")
    print(f"Stacking coefficients: {stack_model.coef_}, intercept: {stack_model.intercept_}")

    test_stack_X = np.column_stack((test_lgb, test_cat, test_xgb))
    final_pred = stack_model.predict(test_stack_X)
    final_pred = np.clip(final_pred, 0.0, 100.0)

    # ------------------------------------------------------------------
    # Save submission
    submission = pd.DataFrame({'Id': test_ids_save, 'Pawpularity': final_pred})
    submission.to_csv(os.path.join(SUBMISSION_DIR, 'submission.csv'), index=False)
    print("\nSubmission saved to submission/submission.csv")
    print(submission.head())

if __name__ == '__main__':
    main()