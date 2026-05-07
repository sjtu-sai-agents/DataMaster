import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import matthews_corrcoef
from sklearn.preprocessing import LabelEncoder
import gc
import os

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
SEED = 42
FRAME_RATE = 59.94
PRE_SNAP_SECONDS = 5.0
STEP_DT = 0.1
WINDOW_SIZE = 5           # for rolling features
RADIUS_1 = 1.0            # yards for congestion
RADIUS_2 = 2.0

def step_to_frame(step):
    """Convert step (0.1 sec) to video frame number."""
    return int(round((step * STEP_DT + PRE_SNAP_SECONDS) * FRAME_RATE))

# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------
def load_data(is_train=True):
    if is_train:
        labels = pd.read_csv('input/train_labels.csv')
        labels['is_ground'] = (labels['nfl_player_id_2'] == 'G')
        labels['contact'] = labels['contact'].astype(int)
        labels['game_play'] = labels['game_play'].astype(str)
        # Convert player IDs to string for consistency
        labels['nfl_player_id_1'] = labels['nfl_player_id_1'].astype(str)
        labels['nfl_player_id_2'] = labels['nfl_player_id_2'].astype(str)
        tracking = pd.read_csv('input/train_player_tracking.csv')
        helmets = pd.read_csv('input/train_baseline_helmets.csv')
    else:
        sub = pd.read_csv('input/sample_submission.csv')
        # parse contact_id: {game_key}_{play_id}_{step}_{player1}_{player2}
        parsed = sub['contact_id'].str.split('_', expand=True)
        sub['game_key'] = parsed[0]
        sub['play_id'] = parsed[1]
        sub['step'] = parsed[2].astype(int)
        sub['nfl_player_id_1'] = parsed[3]
        sub['nfl_player_id_2'] = parsed[4]
        sub['game_play'] = sub['game_key'] + '_' + sub['play_id']
        sub['is_ground'] = (sub['nfl_player_id_2'] == 'G')
        sub['contact'] = -1   # placeholder
        labels = sub
        tracking = pd.read_csv('input/test_player_tracking.csv')
        helmets = pd.read_csv('input/test_baseline_helmets.csv')
    # Convert tracking and helmets IDs to string
    tracking['nfl_player_id'] = tracking['nfl_player_id'].astype(str)
    helmets['nfl_player_id'] = helmets['nfl_player_id'].astype(str)
    return labels, tracking, helmets

# ---------------------------------------------------------------------
# Enhance tracking data with kinematic derivatives
# ---------------------------------------------------------------------
def enhance_tracking(df_tracking):
    cols = ['game_play', 'nfl_player_id', 'step',
            'x_position', 'y_position', 'speed', 'direction',
            'orientation', 'acceleration', 'sa', 'team', 'position']
    df = df_tracking[cols].copy()
    numeric = ['step', 'x_position', 'y_position', 'speed', 'direction',
               'orientation', 'acceleration', 'sa']
    for c in numeric:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df.fillna(0, inplace=True)
    df.sort_values(['game_play', 'nfl_player_id', 'step'], inplace=True)

    # velocity components
    rad = np.deg2rad(df['direction'])
    df['vx'] = df['speed'] * np.sin(rad)
    df['vy'] = df['speed'] * np.cos(rad)

    # acceleration components (derivative of velocity)
    df[['ax', 'ay']] = df.groupby(['game_play', 'nfl_player_id'])[['vx', 'vy']].diff() / STEP_DT
    df[['ax', 'ay']] = df[['ax', 'ay']].fillna(0)

    # angular velocity (absolute change in orientation)
    def angular_diff(s):
        diff = s.diff()
        diff = ((diff + 180) % 360) - 180   # smallest signed difference
        return diff.abs()
    df['angular_vel'] = df.groupby(['game_play', 'nfl_player_id'])['orientation'].transform(angular_diff) / STEP_DT
    df['angular_vel'] = df['angular_vel'].fillna(0)

    return df

# ---------------------------------------------------------------------
# Spatial congestion features (ally/enemy counts within radii)
# ---------------------------------------------------------------------
def compute_congestion(df_tracking):
    sub = df_tracking[['game_play', 'step', 'nfl_player_id',
                       'x_position', 'y_position', 'team']].copy()
    sub.dropna(inplace=True)
    congestion = []
    for (gp, step), grp in sub.groupby(['game_play', 'step']):
        players = grp[['nfl_player_id', 'x_position', 'y_position', 'team']].values
        n = len(players)
        ids = players[:,0]
        x = players[:,1].astype(float)
        y = players[:,2].astype(float)
        team = players[:,3]
        # distance matrix
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        for i in range(n):
            mask1 = (dist[i] <= RADIUS_1) & (np.arange(n) != i)
            mask2 = (dist[i] <= RADIUS_2) & (np.arange(n) != i)
            same_team = (team == team[i])
            ally1 = np.sum(mask1 & same_team)
            enemy1 = np.sum(mask1 & ~same_team)
            ally2 = np.sum(mask2 & same_team)
            enemy2 = np.sum(mask2 & ~same_team)
            congestion.append({
                'game_play': gp,
                'step': step,
                'nfl_player_id': ids[i],
                'ally_count_1y': ally1,
                'enemy_count_1y': enemy1,
                'ally_count_2y': ally2,
                'enemy_count_2y': enemy2
            })
    return pd.DataFrame(congestion)

# ---------------------------------------------------------------------
# Process helmet data: pivot to get per (game_play, frame, player)
# ---------------------------------------------------------------------
def process_helmets(df_helmets):
    df = df_helmets[df_helmets['view'].isin(['Sideline', 'Endzone'])].copy()
    df['center_x'] = df['left'] + df['width'] / 2
    df['center_y'] = df['top'] + df['height'] / 2
    # separate views
    sideline = df[df['view'] == 'Sideline'][['game_play', 'frame', 'nfl_player_id',
                                             'center_x', 'center_y', 'width', 'height']]
    sideline.columns = ['game_play', 'frame', 'nfl_player_id',
                        'sideline_x', 'sideline_y', 'sideline_w', 'sideline_h']
    endzone = df[df['view'] == 'Endzone'][['game_play', 'frame', 'nfl_player_id',
                                           'center_x', 'center_y', 'width', 'height']]
    endzone.columns = ['game_play', 'frame', 'nfl_player_id',
                       'endzone_x', 'endzone_y', 'endzone_w', 'endzone_h']
    # outer merge
    pivot = pd.merge(sideline, endzone, on=['game_play', 'frame', 'nfl_player_id'], how='outer')
    return pivot

# ---------------------------------------------------------------------
# Feature engineering for pairs (train or test)
# ---------------------------------------------------------------------
def engineer_features(df_pairs, df_tracking_feat, df_congestion, helmet_pivot, is_train):
    df = df_pairs.copy()

    # Add frame from step
    df['frame'] = df['step'].apply(step_to_frame)

    # Ensure player IDs are strings for merging
    df['nfl_player_id_1'] = df['nfl_player_id_1'].astype(str)
    df['nfl_player_id_2'] = df['nfl_player_id_2'].astype(str)

    # ---------- Merge tracking for player 1 ----------
    tr1 = df_tracking_feat.copy()
    tr1['nfl_player_id'] = tr1['nfl_player_id'].astype(str)
    tr1.columns = [f'{c}_1' if c not in ['game_play', 'step', 'nfl_player_id'] else c for c in tr1.columns]
    tr1.rename(columns={'nfl_player_id': 'nfl_player_id_1'}, inplace=True)
    df = pd.merge(df, tr1, on=['game_play', 'step', 'nfl_player_id_1'], how='left')

    # ---------- Merge tracking for player 2 ----------
    tr2 = df_tracking_feat.copy()
    tr2['nfl_player_id'] = tr2['nfl_player_id'].astype(str)
    tr2.columns = [f'{c}_2' if c not in ['game_play', 'step', 'nfl_player_id'] else c for c in tr2.columns]
    tr2.rename(columns={'nfl_player_id': 'nfl_player_id_2'}, inplace=True)
    df = pd.merge(df, tr2, on=['game_play', 'step', 'nfl_player_id_2'], how='left')

    # For ground contacts: fill player2 tracking with 0
    ground = df['is_ground']
    for col in tr2.columns:
        if col.endswith('_2') and col not in ['game_play', 'step', 'nfl_player_id_2']:
            df.loc[ground, col] = 0.0
    if 'team_2' in df.columns:
        df.loc[ground, 'team_2'] = 'None'
    if 'position_2' in df.columns:
        df.loc[ground, 'position_2'] = 'Ground'

    # ---------- Merge congestion for player 1 ----------
    cong1 = df_congestion.copy()
    cong1['nfl_player_id'] = cong1['nfl_player_id'].astype(str)
    cong1.columns = [f'{c}_1' if c not in ['game_play', 'step', 'nfl_player_id'] else c for c in cong1.columns]
    cong1.rename(columns={'nfl_player_id': 'nfl_player_id_1'}, inplace=True)
    df = pd.merge(df, cong1, on=['game_play', 'step', 'nfl_player_id_1'], how='left')
    for col in cong1.columns:
        if col.endswith('_1') and col not in ['game_play', 'step', 'nfl_player_id_1']:
            df[col] = df[col].fillna(0)

    # ---------- Merge congestion for player 2 ----------
    cong2 = df_congestion.copy()
    cong2['nfl_player_id'] = cong2['nfl_player_id'].astype(str)
    cong2.columns = [f'{c}_2' if c not in ['game_play', 'step', 'nfl_player_id'] else c for c in cong2.columns]
    cong2.rename(columns={'nfl_player_id': 'nfl_player_id_2'}, inplace=True)
    df = pd.merge(df, cong2, on=['game_play', 'step', 'nfl_player_id_2'], how='left')
    for col in cong2.columns:
        if col.endswith('_2') and col not in ['game_play', 'step', 'nfl_player_id_2']:
            df[col] = df[col].fillna(0)

    # ---------- Merge helmet data for player 1 ----------
    h1 = helmet_pivot.copy()
    h1['nfl_player_id'] = h1['nfl_player_id'].astype(str)
    h1.columns = [f'{c}_1' if c not in ['game_play', 'frame', 'nfl_player_id'] else c for c in h1.columns]
    h1.rename(columns={'nfl_player_id': 'nfl_player_id_1'}, inplace=True)
    df = pd.merge(df, h1, on=['game_play', 'frame', 'nfl_player_id_1'], how='left')

    # ---------- Merge helmet data for player 2 ----------
    h2 = helmet_pivot.copy()
    h2['nfl_player_id'] = h2['nfl_player_id'].astype(str)
    h2.columns = [f'{c}_2' if c not in ['game_play', 'frame', 'nfl_player_id'] else c for c in h2.columns]
    h2.rename(columns={'nfl_player_id': 'nfl_player_id_2'}, inplace=True)
    df = pd.merge(df, h2, on=['game_play', 'frame', 'nfl_player_id_2'], how='left')

    # ---------- Derived pair features ----------
    # Interaction from tracking
    dx = df['x_position_1'] - df['x_position_2']
    dy = df['y_position_1'] - df['y_position_2']
    df['distance'] = np.sqrt(dx**2 + dy**2)
    df.loc[ground, 'distance'] = 0.0

    vx_rel = df['vx_1'] - df['vx_2']
    vy_rel = df['vy_1'] - df['vy_2']
    dot = vx_rel * dx + vy_rel * dy
    dist_safe = df['distance'].copy()
    dist_safe[dist_safe == 0] = 1e-9
    df['closing_speed'] = -dot / dist_safe
    df.loc[ground, 'closing_speed'] = 0.0

    df['rel_speed_mag'] = np.sqrt(vx_rel**2 + vy_rel**2)
    df.loc[ground, 'rel_speed_mag'] = 0.0

    # Geometric alignment
    def angle_to_vector(deg):
        rad = np.deg2rad(deg)
        return np.sin(rad), np.cos(rad)

    o1_sin, o1_cos = angle_to_vector(df['orientation_1'])
    o2_sin, o2_cos = angle_to_vector(df['orientation_2'])
    d1_sin, d1_cos = angle_to_vector(df['direction_1'])
    d2_sin, d2_cos = angle_to_vector(df['direction_2'])

    r_hat_x = dx / dist_safe
    r_hat_y = dy / dist_safe
    r_hat_x[ground] = 0
    r_hat_y[ground] = 0

    df['cos_o1_p2'] = o1_sin * r_hat_x + o1_cos * r_hat_y
    df['cos_d1_p2'] = d1_sin * r_hat_x + d1_cos * r_hat_y
    df['cos_o2_p1'] = -(o2_sin * r_hat_x + o2_cos * r_hat_y)
    df['cos_d2_p1'] = -(d2_sin * r_hat_x + d2_cos * r_hat_y)
    df['cos_o1_o2'] = o1_sin * o2_sin + o1_cos * o2_cos
    df['cos_d1_d2'] = d1_sin * d2_sin + d1_cos * d2_cos
    for col in ['cos_o1_p2', 'cos_d1_p2', 'cos_o2_p1', 'cos_d2_p1', 'cos_o1_o2', 'cos_d1_d2']:
        df.loc[ground, col] = 0.0

    # Helmet pair features for each view
    for view in ['sideline', 'endzone']:
        x1, y1, w1, h1 = f'{view}_x_1', f'{view}_y_1', f'{view}_w_1', f'{view}_h_1'
        x2, y2, w2, h2 = f'{view}_x_2', f'{view}_y_2', f'{view}_w_2', f'{view}_h_2'

        # Distance
        dx_h = df[x1] - df[x2]
        dy_h = df[y1] - df[y2]
        df[f'{view}_dist'] = np.sqrt(dx_h**2 + dy_h**2)

        # IoU
        left1 = df[x1] - df[w1]/2
        right1 = left1 + df[w1]
        top1 = df[y1] - df[h1]/2
        bottom1 = top1 + df[h1]
        left2 = df[x2] - df[w2]/2
        right2 = left2 + df[w2]
        top2 = df[y2] - df[h2]/2
        bottom2 = top2 + df[h2]
        inter_x1 = np.maximum(left1, left2)
        inter_y1 = np.maximum(top1, top2)
        inter_x2 = np.minimum(right1, right2)
        inter_y2 = np.minimum(bottom1, bottom2)
        inter_area = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)
        area1 = df[w1] * df[h1]
        area2 = df[w2] * df[h2]
        union = area1 + area2 - inter_area
        df[f'{view}_iou'] = inter_area / (union + 1e-9)

        # Aspect ratio for player 1
        df[f'{view}_ar_1'] = df[w1] / (df[h1] + 1e-9)

        # Fill missing
        df[f'{view}_dist'] = df[f'{view}_dist'].fillna(9999.0)
        df[f'{view}_iou'] = df[f'{view}_iou'].fillna(0.0)
        df[f'{view}_ar_1'] = df[f'{view}_ar_1'].fillna(1.0)

    # ---------- Temporal features ----------
    df['pair_id'] = df['game_play'] + '_' + df['nfl_player_id_1'] + '_' + df['nfl_player_id_2']
    df = df.sort_values(['pair_id', 'step'])

    base_cols = ['distance', 'closing_speed', 'acceleration_1', 'acceleration_2',
                 'speed_1', 'speed_2', 'sideline_dist', 'endzone_dist']
    # Ensure columns exist
    base_cols = [c for c in base_cols if c in df.columns]

    new_parts = []
    for pid, grp in df.groupby('pair_id'):
        grp = grp.copy()
        for col in base_cols:
            # Rolling stats (window 5, center)
            roll = grp[col].rolling(window=WINDOW_SIZE, min_periods=1, center=True)
            grp[f'{col}_rollmean5'] = roll.mean().values
            grp[f'{col}_rollstd5'] = roll.std().values
            grp[f'{col}_rollmin5'] = roll.min().values
            grp[f'{col}_rollmax5'] = roll.max().values
            grp[f'{col}_sub_min'] = grp[col] - grp[f'{col}_rollmin5']
            grp[f'{col}_max_sub'] = grp[f'{col}_rollmax5'] - grp[col]
            # Lag/Lead
            for shift in [-2, -1, 1, 2]:
                grp[f'{col}_shift{shift}'] = grp[col].shift(-shift).values
        new_parts.append(grp)
    df = pd.concat(new_parts, ignore_index=False).sort_index()

    # Fill NaN from shifts and rolling std
    for col in base_cols:
        for shift in [-2, -1, 1, 2]:
            df[f'{col}_shift{shift}'] = df[f'{col}_shift{shift}'].fillna(0)
        df[f'{col}_rollstd5'] = df[f'{col}_rollstd5'].fillna(0)

    # ---------- Cleanup ----------
    # Drop columns that won't be used as features
    drop_cols = ['frame']  # keep pair_id for smoothing
    if is_train:
        target = df['contact'].copy()
    else:
        target = None
        drop_cols.append('contact')  # placeholder column
    # Fill remaining NaNs
    df.fillna(0, inplace=True)

    # Downcast to save memory
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = df[col].astype('int32')

    return df, target

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    np.random.seed(SEED)

    # -------------------- TRAIN --------------------
    print("Loading training data...")
    train_labels, train_tracking, train_helmets = load_data(is_train=True)

    print("Enhancing tracking...")
    train_tracking_feat = enhance_tracking(train_tracking)
    del train_tracking; gc.collect()

    print("Computing congestion...")
    train_congestion = compute_congestion(train_tracking_feat)

    print("Processing helmets...")
    train_helmet_pivot = process_helmets(train_helmets)
    del train_helmets; gc.collect()

    print("Engineering training features...")
    train_df, train_y = engineer_features(train_labels, train_tracking_feat,
                                          train_congestion, train_helmet_pivot, is_train=True)

    # Encode categorical columns (position, team)
    cat_cols = [c for c in train_df.columns if c.startswith('position_') or c.startswith('team_')]
    cat_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        train_df[col] = le.fit_transform(train_df[col].astype(str))
        cat_encoders[col] = le

    # Separate features/target/groups
    cols_to_drop = [
        'contact', 'game_play', 'step', 'nfl_player_id_1', 'nfl_player_id_2', 'is_ground',
        'contact_id', 'datetime', 'pair_id'
    ]
    drop_cols = [c for c in cols_to_drop if c in train_df.columns]
    X = train_df.drop(columns=drop_cols)
    y = train_y
    groups = train_df['game_play']
    is_ground = train_df['is_ground']

    # Store feature list to align test data later
    feature_cols = X.columns.tolist()

    X_pp = X[~is_ground]
    y_pp = y[~is_ground]
    groups_pp = groups[~is_ground]
    X_pg = X[is_ground]
    y_pg = y[is_ground]
    groups_pg = groups[is_ground]

    # -------------------- CROSS-VALIDATION --------------------
    gkf = GroupKFold(n_splits=5)

    # Player-Player
    oof_pp = np.zeros(len(X_pp))
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_pp, groups=groups_pp)):
        print(f"PP Fold {fold+1}")
        X_tr, X_va = X_pp.iloc[train_idx], X_pp.iloc[val_idx]
        y_tr, y_va = y_pp.iloc[train_idx], y_pp.iloc[val_idx]
        model = xgb.XGBClassifier(n_estimators=2000, learning_rate=0.05, max_depth=9,
                                  subsample=0.8, colsample_bytree=0.8, objective='binary:logistic',
                                  tree_method='gpu_hist', random_state=SEED, n_jobs=-1,
                                  early_stopping_rounds=50)  # moved to constructor
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=100)
        oof_pp[val_idx] = model.predict_proba(X_va)[:, 1]

    # Player-Ground
    oof_pg = np.zeros(len(X_pg))
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_pg, groups=groups_pg)):
        print(f"PG Fold {fold+1}")
        X_tr, X_va = X_pg.iloc[train_idx], X_pg.iloc[val_idx]
        y_tr, y_va = y_pg.iloc[train_idx], y_pg.iloc[val_idx]
        model = xgb.XGBClassifier(n_estimators=2000, learning_rate=0.05, max_depth=9,
                                  subsample=0.8, colsample_bytree=0.8, objective='binary:logistic',
                                  tree_method='gpu_hist', random_state=SEED, n_jobs=-1,
                                  early_stopping_rounds=50)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=100)
        oof_pg[val_idx] = model.predict_proba(X_va)[:, 1]

    # Threshold optimization
    best_th_pp, best_mcc_pp = 0.5, -1
    for th in np.arange(0.1, 0.9, 0.01):
        mcc = matthews_corrcoef(y_pp, (oof_pp >= th).astype(int))
        if mcc > best_mcc_pp:
            best_mcc_pp = mcc
            best_th_pp = th
    print(f"PP OOF MCC = {best_mcc_pp:.4f} (th={best_th_pp:.2f})")

    best_th_pg, best_mcc_pg = 0.5, -1
    for th in np.arange(0.1, 0.9, 0.01):
        mcc = matthews_corrcoef(y_pg, (oof_pg >= th).astype(int))
        if mcc > best_mcc_pg:
            best_mcc_pg = mcc
            best_th_pg = th
    print(f"PG OOF MCC = {best_mcc_pg:.4f} (th={best_th_pg:.2f})")

    # Combined OOF
    oof_all = np.zeros(len(y))
    oof_all[~is_ground] = oof_pp
    oof_all[is_ground] = oof_pg
    pred_all = np.zeros_like(oof_all, dtype=int)
    pred_all[~is_ground] = (oof_pp >= best_th_pp).astype(int)
    pred_all[is_ground] = (oof_pg >= best_th_pg).astype(int)
    overall_mcc = matthews_corrcoef(y, pred_all)
    print(f"Overall OOF MCC = {overall_mcc:.4f}")

    # -------------------- FINAL MODELS --------------------
    print("Training final PP model...")
    model_pp = xgb.XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=9,
                                 subsample=0.8, colsample_bytree=0.8, objective='binary:logistic',
                                 tree_method='gpu_hist', random_state=SEED, n_jobs=-1)
    model_pp.fit(X_pp, y_pp)

    print("Training final PG model...")
    model_pg = xgb.XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=9,
                                 subsample=0.8, colsample_bytree=0.8, objective='binary:logistic',
                                 tree_method='gpu_hist', random_state=SEED, n_jobs=-1)
    model_pg.fit(X_pg, y_pg)

    # -------------------- TEST --------------------
    print("Loading test data...")
    test_labels, test_tracking, test_helmets = load_data(is_train=False)

    print("Enhancing test tracking...")
    test_tracking_feat = enhance_tracking(test_tracking)
    del test_tracking; gc.collect()

    print("Computing test congestion...")
    test_congestion = compute_congestion(test_tracking_feat)

    print("Processing test helmets...")
    test_helmet_pivot = process_helmets(test_helmets)
    del test_helmets; gc.collect()

    print("Engineering test features...")
    test_df, _ = engineer_features(test_labels, test_tracking_feat,
                                   test_congestion, test_helmet_pivot, is_train=False)

    # Encode categorical using saved encoders
    for col in cat_cols:
        if col in test_df.columns:
            le = cat_encoders[col]
            test_df[col] = test_df[col].astype(str).map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
            test_df[col] = test_df[col].replace(-1, 0)

    # Prepare feature matrix (drop same columns as training)
    drop_cols_test = [c for c in cols_to_drop if c in test_df.columns]
    X_test = test_df.drop(columns=drop_cols_test)

    # Align with training feature set
    X_test = X_test.reindex(columns=feature_cols, fill_value=0)

    is_ground_test = test_df['is_ground']

    X_test_pp = X_test[~is_ground_test]
    X_test_pg = X_test[is_ground_test]

    # Predict
    pp_probs = model_pp.predict_proba(X_test_pp)[:, 1]
    pg_probs = model_pg.predict_proba(X_test_pg)[:, 1]

    # Combine and smooth
    test_probs = np.zeros(len(test_df))
    test_probs[~is_ground_test] = pp_probs
    test_probs[is_ground_test] = pg_probs

    # Smooth with 5-step rolling mean per pair
    test_df_sorted = test_df.sort_values(['pair_id', 'step'])
    temp = test_df_sorted[['pair_id']].copy()
    temp['prob'] = test_probs[test_df_sorted.index]
    temp['prob_smooth'] = temp.groupby('pair_id')['prob'].transform(
        lambda s: s.rolling(window=5, min_periods=1, center=True).mean())
    temp['prob_smooth'] = temp['prob_smooth'].fillna(temp['prob'])
    test_probs_smooth = temp['prob_smooth'].values[np.argsort(test_df_sorted.index)]

    # Apply thresholds
    final_pred = np.zeros(len(test_probs_smooth), dtype=int)
    final_pred[~is_ground_test] = (test_probs_smooth[~is_ground_test] >= best_th_pp).astype(int)
    final_pred[is_ground_test] = (test_probs_smooth[is_ground_test] >= best_th_pg).astype(int)

    # Create submission
    submission = test_labels[['contact_id']].copy()
    submission['contact'] = final_pred
    os.makedirs('submission', exist_ok=True)
    submission.to_csv('submission/submission.csv', index=False)
    print("Submission saved to submission/submission.csv")
    print(f"Validation MCC (overall OOF): {overall_mcc:.4f}")

if __name__ == "__main__":
    main()