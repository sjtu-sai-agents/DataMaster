import os
import json
import random
import numpy as np
import pandas as pd
from typing import List, Dict, Any

# Constants
K = 5
XY_WEIGHT = 1.0
YAW_WEIGHT = 10.0
FEAT_WEIGHT = 5.0
MERGE_THRESH = 3.0
VAL_FRAC = 0.2
THRESHOLDS = np.arange(0.5, 1.0, 0.05)  # 0.5,0.55,...,0.95

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def quat_to_yaw(q):
    """Convert quaternion [w,x,y,z] to yaw around Z axis."""
    w, x, y, z = q
    yaw = np.arctan2(2.0*(w*z + x*y), 1.0 - 2.0*(y*y + z*z))
    return yaw

def normalize_angle(angle):
    """Normalize angle to [-pi, pi]."""
    while angle > np.pi:
        angle -= 2*np.pi
    while angle <= -np.pi:
        angle += 2*np.pi
    return angle

def rot2d(theta):
    """2D rotation matrix."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])

def build_sample_to_pose(data_dir):
    """Map sample_token -> {'t': [x,y,z], 'yaw': float} using first sample_data."""
    ego_poses = load_json(os.path.join(data_dir, 'ego_pose.json'))
    pose_map = {p['token']: {'t': np.array(p['translation']), 'yaw': quat_to_yaw(p['rotation'])} for p in ego_poses}

    sample_data = load_json(os.path.join(data_dir, 'sample_data.json'))
    sample_to_ego = {}
    for sd in sample_data:
        sample_token = sd.get('sample_token')
        ego_token = sd.get('ego_pose_token')
        if sample_token and ego_token and sample_token not in sample_to_ego:
            sample_to_ego[sample_token] = ego_token

    sample_to_pose = {}
    for sample_token, ego_token in sample_to_ego.items():
        if ego_token in pose_map:
            sample_to_pose[sample_token] = pose_map[ego_token]
    return sample_to_pose

def parse_train_annotations(csv_path):
    """Parse train.csv into dict: sample_id -> list of box dicts."""
    df = pd.read_csv(csv_path)
    id_to_boxes = {}
    for _, row in df.iterrows():
        sample_id = row['Id']
        pred_str = row['PredictionString']
        boxes = []
        if isinstance(pred_str, str) and pred_str.strip():
            tokens = pred_str.split()
            for i in range(0, len(tokens), 8):
                cx, cy, cz, w, l, h, yaw, cls = tokens[i:i+8]
                box = {
                    'cx': float(cx), 'cy': float(cy), 'cz': float(cz),
                    'w': float(w), 'l': float(l), 'h': float(h),
                    'yaw': float(yaw), 'cls': cls
                }
                boxes.append(box)
        id_to_boxes[sample_id] = boxes
    return id_to_boxes

def build_scene_feature_maps(data_dir):
    """Map sample_token -> 6-dim binary feature from scene description."""
    sample_path = os.path.join(data_dir, 'sample.json')
    scene_path = os.path.join(data_dir, 'scene.json')
    if not os.path.exists(scene_path) or not os.path.exists(sample_path):
        return {}
    sample_json = load_json(sample_path)
    scene_json = load_json(scene_path)
    scene_to_desc = {s['token']: s['description'] for s in scene_json}
    sample_to_scene = {s['token']: s['scene_token'] for s in sample_json}
    keywords = ["intersection", "turn", "stop", "yield", "residential", "parking"]
    sample_to_feat = {}
    for sample_token, scene_token in sample_to_scene.items():
        desc = scene_to_desc.get(scene_token, "").lower()
        feat = [1 if kw in desc else 0 for kw in keywords]
        sample_to_feat[sample_token] = feat
    return sample_to_feat

def world_to_ego_box(box, ego_t, ego_yaw):
    """Convert box from world coordinates to ego-relative coordinates."""
    dx = box['cx'] - ego_t[0]
    dy = box['cy'] - ego_t[1]
    dz = box['cz'] - ego_t[2]
    R = rot2d(-ego_yaw)
    rel_xy = R @ np.array([dx, dy])
    ryaw = normalize_angle(box['yaw'] - ego_yaw)
    return {
        'cx': rel_xy[0], 'cy': rel_xy[1], 'cz': dz,
        'w': box['w'], 'l': box['l'], 'h': box['h'],
        'yaw': ryaw, 'cls': box['cls']
    }

def ego_to_world_box(relbox, ego_t, ego_yaw):
    """Convert box from ego-relative coordinates to world coordinates."""
    R = rot2d(ego_yaw)
    world_xy = R @ np.array([relbox['cx'], relbox['cy']])
    cx = world_xy[0] + ego_t[0]
    cy = world_xy[1] + ego_t[1]
    cz = relbox['cz'] + ego_t[2]
    yaw = normalize_angle(relbox['yaw'] + ego_yaw)
    return {
        'cx': cx, 'cy': cy, 'cz': cz,
        'w': relbox['w'], 'l': relbox['l'], 'h': relbox['h'],
        'yaw': yaw, 'cls': relbox['cls']
    }

def build_db(sample_ids, id_to_boxes, sample_to_pose, sample_to_features):
    """Build database of training samples for fast lookup."""
    db_tokens = []
    db_xy = []
    db_yaw = []
    db_feat = []
    db_rel_boxes = []
    for sid in sample_ids:
        pose = sample_to_pose.get(sid)
        if pose is None:
            continue
        boxes = id_to_boxes.get(sid, [])
        feat = sample_to_features.get(sid, [0]*6)
        rel_boxes = [world_to_ego_box(b, pose['t'], pose['yaw']) for b in boxes]
        db_tokens.append(sid)
        db_xy.append(pose['t'][:2])
        db_yaw.append(pose['yaw'])
        db_feat.append(feat)
        db_rel_boxes.append(rel_boxes)
    return {
        'tokens': db_tokens,
        'xy': np.array(db_xy),
        'yaw': np.array(db_yaw),
        'feat': np.array(db_feat),
        'rel_boxes': db_rel_boxes
    }

def compute_distances(query_xy, query_yaw, query_feat, db_xy, db_yaw, db_feat,
                      xy_weight, yaw_weight, feat_weight):
    """Vectorized distance computation between query and all DB entries."""
    dxy = np.sqrt(np.sum((db_xy - query_xy) ** 2, axis=1))
    dyaw = np.abs(np.arctan2(np.sin(db_yaw - query_yaw), np.cos(db_yaw - query_yaw)))
    dfeat = np.sqrt(np.sum((db_feat - query_feat) ** 2, axis=1))
    return xy_weight * dxy + yaw_weight * dyaw + feat_weight * dfeat

def predict_for_sample(query_token, query_pose, query_feat, db, k=K,
                       merge_thresh=MERGE_THRESH,
                       xy_weight=XY_WEIGHT, yaw_weight=YAW_WEIGHT,
                       feat_weight=FEAT_WEIGHT):
    """Predict boxes for a query sample using kNN and merging."""
    if query_pose is None:
        return []
    query_xy = query_pose['t'][:2]
    query_yaw = query_pose['yaw']
    query_feat = np.array(query_feat)

    dists = compute_distances(query_xy, query_yaw, query_feat,
                              db['xy'], db['yaw'], db['feat'],
                              xy_weight, yaw_weight, feat_weight)

    if len(dists) <= k:
        idxs = np.argsort(dists)
    else:
        idxs = np.argpartition(dists, k)[:k]
        idxs = idxs[np.argsort(dists[idxs])]

    candidates = []  # each: {'box': world_box, 'weight': float, 'dist': float}
    for i in idxs:
        d = dists[i]
        weight = 1.0 / (d + 1e-6)
        for relbox in db['rel_boxes'][i]:
            world_box = ego_to_world_box(relbox, query_pose['t'], query_pose['yaw'])
            candidates.append({'box': world_box, 'weight': weight, 'dist': d})

    if not candidates:
        return []

    # Group by class
    class_groups = {}
    for cand in candidates:
        cls = cand['box']['cls']
        class_groups.setdefault(cls, []).append(cand)

    merged_preds = []
    for cls, items in class_groups.items():
        centers = np.array([[c['box']['cx'], c['box']['cy'], c['box']['cz']] for c in items])
        weights = np.array([c['weight'] for c in items])
        dists_neigh = np.array([c['dist'] for c in items])
        n = len(items)
        assigned = [False] * n

        for i in range(n):
            if assigned[i]:
                continue
            cluster_idx = [i]
            assigned[i] = True
            queue = [i]
            while queue:
                cur = queue.pop()
                for j in range(n):
                    if not assigned[j]:
                        if np.linalg.norm(centers[cur] - centers[j]) < merge_thresh:
                            cluster_idx.append(j)
                            assigned[j] = True
                            queue.append(j)
            # Merge cluster
            total_weight = weights[cluster_idx].sum()
            min_dist = dists_neigh[cluster_idx].min()
            cx = np.average(centers[cluster_idx, 0], weights=weights[cluster_idx])
            cy = np.average(centers[cluster_idx, 1], weights=weights[cluster_idx])
            cz = np.average(centers[cluster_idx, 2], weights=weights[cluster_idx])
            w_arr = np.array([items[idx]['box']['w'] for idx in cluster_idx])
            l_arr = np.array([items[idx]['box']['l'] for idx in cluster_idx])
            h_arr = np.array([items[idx]['box']['h'] for idx in cluster_idx])
            yaw_arr = np.array([items[idx]['box']['yaw'] for idx in cluster_idx])
            w_avg = np.average(w_arr, weights=weights[cluster_idx])
            l_avg = np.average(l_arr, weights=weights[cluster_idx])
            h_avg = np.average(h_arr, weights=weights[cluster_idx])
            sin_yaw = np.average(np.sin(yaw_arr), weights=weights[cluster_idx])
            cos_yaw = np.average(np.cos(yaw_arr), weights=weights[cluster_idx])
            yaw_avg = np.arctan2(sin_yaw, cos_yaw)
            confidence = total_weight * np.exp(-min_dist / 10.0)
            merged_preds.append({
                'cx': cx, 'cy': cy, 'cz': cz,
                'w': w_avg, 'l': l_avg, 'h': h_avg,
                'yaw': yaw_avg, 'cls': cls, 'confidence': confidence
            })

    merged_preds.sort(key=lambda x: x['confidence'], reverse=True)
    return merged_preds

def compute_iou_aa(box1, box2):
    """Axis-aligned 3D IoU (ignores orientation)."""
    x1_min, x1_max = box1['cx'] - box1['w']/2, box1['cx'] + box1['w']/2
    y1_min, y1_max = box1['cy'] - box1['l']/2, box1['cy'] + box1['l']/2
    z1_min, z1_max = box1['cz'] - box1['h']/2, box1['cz'] + box1['h']/2

    x2_min, x2_max = box2['cx'] - box2['w']/2, box2['cx'] + box2['w']/2
    y2_min, y2_max = box2['cy'] - box2['l']/2, box2['cy'] + box2['l']/2
    z2_min, z2_max = box2['cz'] - box2['h']/2, box2['cz'] + box2['h']/2

    inter_x = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
    inter_y = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
    inter_z = max(0, min(z1_max, z2_max) - max(z1_min, z2_min))
    inter_vol = inter_x * inter_y * inter_z

    vol1 = box1['w'] * box1['l'] * box1['h']
    vol2 = box2['w'] * box2['l'] * box2['h']
    union_vol = vol1 + vol2 - inter_vol
    return inter_vol / union_vol if union_vol > 0 else 0.0

def compute_ap_per_image(gt_boxes, pred_boxes, thresholds=THRESHOLDS):
    """Average Precision per image as defined in competition."""
    if len(gt_boxes) == 0:
        return 1.0 if len(pred_boxes) == 0 else 0.0

    precisions = []
    for thr in thresholds:
        sorted_preds = sorted(pred_boxes, key=lambda x: x['confidence'], reverse=True)
        gt_used = [False] * len(gt_boxes)
        tp = fp = 0
        for p in sorted_preds:
            best_iou = 0.0
            best_idx = -1
            for i, g in enumerate(gt_boxes):
                if gt_used[i] or p['cls'] != g['cls']:
                    continue
                iou = compute_iou_aa(p, g)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
            if best_iou >= thr:
                tp += 1
                gt_used[best_idx] = True
            else:
                fp += 1
        fn = sum(1 for used in gt_used if not used)
        denom = tp + fp + fn
        prec = tp / denom if denom > 0 else 0.0
        precisions.append(prec)
    return np.mean(precisions)

def format_prediction_string(preds):
    """Convert list of prediction dicts to space-delimited string."""
    if not preds:
        return ''
    parts = []
    for p in preds:
        parts.append(f"{p['confidence']} {p['cx']} {p['cy']} {p['cz']} {p['w']} {p['l']} {p['h']} {p['yaw']} {p['cls']}")
    return ' '.join(parts)

def write_submission(test_ids, preds_dict, out_path):
    """Write submission CSV."""
    with open(out_path, 'w') as f:
        f.write("Id,PredictionString\n")
        for sid in test_ids:
            pred_str = format_prediction_string(preds_dict.get(sid, []))
            f.write(f"{sid},{pred_str}\n")

def split_by_scenes(sample_ids, train_data_dir, val_frac=VAL_FRAC):
    """Split train samples by scene, returning train/val lists."""
    sample_json = load_json(os.path.join(train_data_dir, 'sample.json'))
    sample_to_scene = {}
    sample_set = set(sample_ids)
    for s in sample_json:
        token = s['token']
        if token in sample_set:
            sample_to_scene[token] = s['scene_token']
    scene_to_samples = {}
    for sid, scene_tok in sample_to_scene.items():
        scene_to_samples.setdefault(scene_tok, []).append(sid)

    scene_list = list(scene_to_samples.keys())
    random.shuffle(scene_list)
    n_val = int(val_frac * len(scene_list))
    val_scenes = set(scene_list[:n_val])

    train_sids = [sid for sid in sample_ids if sample_to_scene.get(sid) not in val_scenes]
    val_sids = [sid for sid in sample_ids if sample_to_scene.get(sid) in val_scenes]
    return train_sids, val_sids

def main():
    set_seeds(42)
    input_dir = "./input"
    train_data_dir = os.path.join(input_dir, "train_data")
    test_data_dir = os.path.join(input_dir, "test_data")
    train_csv = os.path.join(input_dir, "train.csv")
    sample_sub = os.path.join(input_dir, "sample_submission.csv")
    out_submission = "./submission/submission.csv"

    os.makedirs("./submission", exist_ok=True)
    os.makedirs("./working", exist_ok=True)

    print("Loading data...")
    id_to_boxes = parse_train_annotations(train_csv)
    all_train_ids = list(id_to_boxes.keys())
    print(f"Number of train samples: {len(all_train_ids)}")

    train_sample_to_pose = build_sample_to_pose(train_data_dir)
    train_sample_to_features = build_scene_feature_maps(train_data_dir)

    print("Splitting by scenes...")
    train_ids, val_ids = split_by_scenes(all_train_ids, train_data_dir, VAL_FRAC)
    print(f"Train split: {len(train_ids)} samples, Val split: {len(val_ids)} samples")

    print("Building database from training split...")
    db = build_db(train_ids, id_to_boxes, train_sample_to_pose, train_sample_to_features)

    print("Evaluating on validation set...")
    val_aps = []
    for i, sid in enumerate(val_ids):
        if i % 100 == 0:
            print(f"Processed {i}/{len(val_ids)} val samples")
        pose = train_sample_to_pose.get(sid)
        feat = train_sample_to_features.get(sid, [0]*6)
        pred_boxes = predict_for_sample(sid, pose, feat, db)
        gt_boxes = id_to_boxes.get(sid, [])
        val_aps.append(compute_ap_per_image(gt_boxes, pred_boxes))
    val_map = np.mean(val_aps)
    print(f"Validation mAP: {val_map:.4f}")

    print("Rebuilding database from full training set...")
    full_db = build_db(all_train_ids, id_to_boxes, train_sample_to_pose, train_sample_to_features)

    test_df = pd.read_csv(sample_sub)
    test_ids = test_df['Id'].tolist()
    print(f"Number of test samples: {len(test_ids)}")

    test_sample_to_pose = build_sample_to_pose(test_data_dir)
    test_sample_to_features = build_scene_feature_maps(test_data_dir)

    test_preds = {}
    for i, sid in enumerate(test_ids):
        if i % 100 == 0:
            print(f"Processed {i}/{len(test_ids)} test samples")
        pose = test_sample_to_pose.get(sid)
        feat = test_sample_to_features.get(sid, [0]*6)
        test_preds[sid] = predict_for_sample(sid, pose, feat, full_db)

    write_submission(test_ids, test_preds, out_submission)
    print(f"Submission written to {out_submission}")

if __name__ == "__main__":
    main()