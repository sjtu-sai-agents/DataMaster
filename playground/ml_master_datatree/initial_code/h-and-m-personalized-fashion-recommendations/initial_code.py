import pandas as pd
import numpy as np
import os
from collections import defaultdict, Counter

# Constants
POP_DAYS = 14
CO_DAYS = 28
REP_DAYS = 28
VAL_DAYS = 6   # number of days held out for validation

def compute_features(trans, age_map):
    """
    trans : DataFrame with columns t_dat, customer_id, article_id
    age_map: dict {customer_id: age_bin (0..6)}
    Returns:
        global_top12: list of 12 article_ids
        age_top: dict {age_bin: list of 12 article_ids}
        co_occ: dict {article_id: list of up to 3 co‑occurring article_ids}
        personal: dict {customer_id: list of article_ids ordered by recency}
    """
    max_date = trans['t_dat'].max()
    
    # ---------- Global & Age Popularity (last POP_DAYS) ----------
    start_pop = max_date - pd.Timedelta(days=POP_DAYS-1)
    trans_pop = trans[trans['t_dat'] >= start_pop].copy()
    trans_pop['days_diff'] = (max_date - trans_pop['t_dat']).dt.days
    trans_pop['weight'] = 1.0 / (trans_pop['days_diff'] + 1.0)
    
    # Global top 12
    global_top = (trans_pop.groupby('article_id')['weight'].sum()
                  .sort_values(ascending=False).head(12).index.tolist())
    
    # Age top 12 per bin
    trans_pop['age_bin'] = trans_pop['customer_id'].map(age_map).fillna(0).astype(int)
    age_pop = trans_pop.groupby(['age_bin', 'article_id'])['weight'].sum()
    age_top = {}
    for age_bin in range(7):  # bins 0..6
        if age_bin in age_pop.index.get_level_values(0):
            top = (age_pop.xs(age_bin, level=0)
                    .sort_values(ascending=False).head(12).index.tolist())
        else:
            top = global_top  # fallback
        age_top[age_bin] = top
    
    # ---------- Co‑occurrence (last CO_DAYS) ----------
    start_co = max_date - pd.Timedelta(days=CO_DAYS-1)
    trans_co = trans[trans['t_dat'] >= start_co].copy()
    co_dict = defaultdict(Counter)
    
    # Iterate over baskets (customer-day)
    for (cust, date), group in trans_co.groupby(['customer_id', 't_dat']):
        articles = group['article_id'].unique()
        if len(articles) < 2:
            continue
        for i in range(len(articles)):
            a = articles[i]
            for j in range(i+1, len(articles)):
                b = articles[j]
                co_dict[a][b] += 1
                co_dict[b][a] += 1
    
    co_occ = {}
    for a, counter in co_dict.items():
        top = [b for b, _ in counter.most_common(3)]
        co_occ[a] = top
    
    # ---------- Personal repurchase history (last REP_DAYS) ----------
    start_rep = max_date - pd.Timedelta(days=REP_DAYS-1)
    trans_rep = trans[trans['t_dat'] >= start_rep].copy()
    # Sort to have most recent purchase first per customer
    trans_rep = trans_rep.sort_values(by=['customer_id', 't_dat'],
                                      ascending=[True, False])
    # Keep only the most recent occurrence of each (customer, article)
    trans_rep = trans_rep.drop_duplicates(subset=['customer_id', 'article_id'],
                                          keep='first')
    personal = trans_rep.groupby('customer_id')['article_id'].apply(list).to_dict()
    
    return global_top, age_top, co_occ, personal

def generate_predictions(customer_list, global_top, age_top, co_occ, personal, age_map):
    """
    Returns dict {customer_id: list of 12 article_ids}
    """
    preds_dict = {}
    for cid in customer_list:
        age_bin = age_map.get(cid, 0)
        seen = set()
        preds = []
        
        # 1. Personal repurchase history (most recent first)
        for a in personal.get(cid, []):
            if a not in seen:
                seen.add(a)
                preds.append(a)
            if len(preds) >= 12:
                break
        
        # 2. Co‑occurrence from up to 3 most recent personal items
        if len(preds) < 12:
            recent_items = personal.get(cid, [])[:3]
            for a in recent_items:
                for b in co_occ.get(a, []):
                    if b not in seen:
                        seen.add(b)
                        preds.append(b)
                        if len(preds) >= 12:
                            break
                if len(preds) >= 12:
                    break
        
        # 3. Age‑group popularity
        if len(preds) < 12:
            for a in age_top[age_bin]:
                if a not in seen:
                    seen.add(a)
                    preds.append(a)
                    if len(preds) >= 12:
                        break
        
        # 4. Global popularity (fallback)
        if len(preds) < 12:
            for a in global_top:
                if a not in seen:
                    seen.add(a)
                    preds.append(a)
                    if len(preds) >= 12:
                        break
        
        # Should always have exactly 12
        preds_dict[cid] = preds[:12]
    return preds_dict

def mapk(actual_dict, pred_dict, k=12):
    """
    Mean Average Precision @ k
    actual_dict: {customer_id: list of true article_ids}
    pred_dict:   {customer_id: list of predicted article_ids (order matters)}
    """
    ap_sum = 0.0
    cnt = 0
    for cid, actual in actual_dict.items():
        if cid not in pred_dict:
            continue
        pred = pred_dict[cid][:k]
        if not actual:
            continue
        actual_set = set(actual)
        score = 0.0
        hits = 0
        for i, p in enumerate(pred):
            if p in actual_set and p not in pred[:i]:  # no duplicate in pred
                hits += 1
                score += hits / (i + 1.0)
        denom = min(len(actual), k)
        ap_sum += score / denom
        cnt += 1
    return ap_sum / cnt if cnt else 0.0

def main():
    print("Loading data...")
    # Load transactions (only needed columns)
    trans = pd.read_csv('input/transactions_train.csv',
                        usecols=['t_dat', 'customer_id', 'article_id'],
                        dtype={'customer_id': 'str', 'article_id': 'str'},
                        parse_dates=['t_dat'])
    
    # Load customers (age only)
    customers = pd.read_csv('input/customers.csv',
                            usecols=['customer_id', 'age'],
                            dtype={'customer_id': 'str'})
    customers['age'] = customers['age'].fillna(-1)
    bins = [-2, 0, 20, 30, 40, 50, 60, 100]
    labels = [0, 1, 2, 3, 4, 5, 6]
    customers['age_bin'] = pd.cut(customers['age'], bins=bins, labels=labels,
                                  include_lowest=True)
    customers['age_bin'] = customers['age_bin'].astype(int)
    age_map = dict(zip(customers['customer_id'], customers['age_bin']))
    
    # Load sample submission to get complete list of customers for final output
    sample_sub = pd.read_csv('input/sample_submission.csv',
                             dtype={'customer_id': 'str'})
    all_customers = sample_sub['customer_id'].tolist()
    
    # ---------- Validation split ----------
    max_date = trans['t_dat'].max()
    val_start = max_date - pd.Timedelta(days=VAL_DAYS-1)   # last VAL_DAYS days inclusive
    train = trans[trans['t_dat'] < val_start].copy()
    val = trans[trans['t_dat'] >= val_start].copy()
    
    print("Computing features on training data...")
    global_top_train, age_top_train, co_occ_train, personal_train = compute_features(train, age_map)
    
    # Ground truth for validation (only customers who bought something)
    val_true = val.groupby('customer_id')['article_id'].apply(lambda x: list(set(x))).to_dict()
    val_customers = list(val_true.keys())
    
    print("Generating predictions for validation customers...")
    val_pred = generate_predictions(val_customers,
                                    global_top_train, age_top_train,
                                    co_occ_train, personal_train, age_map)
    
    map_val = mapk(val_true, val_pred)
    print(f"Validation MAP@12: {map_val:.6f}")
    
    # ---------- Final model on all data ----------
    print("Retraining on full data...")
    global_top_full, age_top_full, co_occ_full, personal_full = compute_features(trans, age_map)
    
    print("Generating final predictions for all customers...")
    final_pred = generate_predictions(all_customers,
                                      global_top_full, age_top_full,
                                      co_occ_full, personal_full, age_map)
    
    # Write submission
    os.makedirs('submission', exist_ok=True)
    sub_df = pd.DataFrame({'customer_id': all_customers})
    sub_df['prediction'] = [' '.join(final_pred[cid]) for cid in all_customers]
    sub_df.to_csv('submission/submission.csv', index=False)
    print("Submission saved to submission/submission.csv")

if __name__ == '__main__':
    main()