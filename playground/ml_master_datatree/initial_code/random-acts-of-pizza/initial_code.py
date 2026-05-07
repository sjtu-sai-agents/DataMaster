import os
import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from scipy.sparse import hstack

# ---------- 1. 快速数据加载与预处理 ----------
print("Loading data...")
train = pd.read_json('./input/train/train.json')
test = pd.read_json('./input/test/test.json')
y = train['requester_received_pizza'].astype(int)

# 合并文本
train['text'] = (train['request_title'] + " " + train['request_text_edit_aware']).fillna("")
test['text'] = (test['request_title'] + " " + test['request_text_edit_aware']).fillna("")

# ---------- 2. 向量化特征提取 (极速) ----------
def fast_features(df):
    feat = pd.DataFrame()
    # 长度特征
    feat['len_text'] = df['text'].str.len()
    feat['len_title'] = df['request_title'].fillna("").str.len()
    # 计数特征 (利用向量化函数)
    feat['excl_count'] = df['text'].str.count('!')
    feat['ques_count'] = df['text'].str.count('\?')
    feat['upper_ratio'] = df['text'].str.findall(r'[A-Z]').str.len() / (feat['len_text'] + 1)
    
    # 时间特征
    dt = pd.to_datetime(df['unix_timestamp_of_request_utc'], unit='s')
    feat['hour'] = dt.dt.hour
    feat['dayofweek'] = dt.dt.dayofweek
    
    # 核心行为指标
    eps = 1e-6
    feat['age'] = df['requester_account_age_in_days_at_request']
    feat['prev_raop_posts'] = df['requester_number_of_posts_on_raop_at_request']
    feat['upvotes_ratio'] = df['requester_upvotes_plus_downvotes_at_request'] / (df['requester_number_of_comments_at_request'] + eps)
    
    return feat

print("Extracting features...")
X_train_meta = fast_features(train)
X_test_meta = fast_features(test)

# ---------- 3. 优化的 TF-IDF ----------
print("Fitting TF-IDF...")
# 减少 max_features 并移除 ngram (1,2) 显著提升速度
tfidfv = TfidfVectorizer(max_features=2000, stop_words='english', min_df=3)
X_train_tfidf = tfidfv.fit_transform(train['text'])
X_test_tfidf = tfidfv.transform(test['text'])

# 合并
X_train = hstack([X_train_tfidf, X_train_meta.values]).tocsr()
X_test = hstack([X_test_tfidf, X_test_meta.values]).tocsr()

# ---------- 4. 轻量级 LightGBM 训练 ----------
print("Starting CV...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
test_preds = np.zeros(len(test))

params = {
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.08, # 略微调高步长
    'num_leaves': 15,       # 减小树的深度，防止过拟合也提速
    'verbose': -1,
    'n_jobs': -1
}

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y)):
    lgb_tr = lgb.Dataset(X_train[tr_idx], label=y.iloc[tr_idx])
    lgb_val = lgb.Dataset(X_train[val_idx], label=y.iloc[val_idx], reference=lgb_tr)
    
    # 简化训练过程
    model = lgb.train(params, lgb_tr, num_boost_round=500,
                      valid_sets=[lgb_val],
                      callbacks=[lgb.early_stopping(25)])
    
    test_preds += model.predict(X_test) / 5

# ---------- 5. 保存结果 ----------
os.makedirs('./submission', exist_ok=True)
pd.DataFrame({'request_id': test['request_id'], 'requester_received_pizza': test_preds}).to_csv('./submission/submission.csv', index=False)
print("Done!")