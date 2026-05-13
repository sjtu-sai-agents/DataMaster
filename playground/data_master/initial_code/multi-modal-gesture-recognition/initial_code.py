import os
import sys
import io
import tarfile
import zipfile
import pickle
import random
from collections import Counter

import numpy as np
import pandas as pd
import scipy.io
import scipy.interpolate
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from tqdm import tqdm

# ----------------------------------------------------------------------
# Constants & Helper Mappings
# ----------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

GESTURE_NAMES = [
    'vattene', 'vieniqui', 'perfetto', 'furbo', 'cheduepalle',
    'chevuoi', 'daccordo', 'seipazzo', 'combinato', 'freganiente',
    'ok', 'cosatifarei', 'basta', 'prendere', 'noncenepiu',
    'fame', 'tantotempo', 'buonissimo', 'messidaccordo', 'sonostufo'
]
GESTURE_TO_ID = {name: i+1 for i, name in enumerate(GESTURE_NAMES)}   # 1..20

JOINT_NAMES = [
    'HipCenter', 'Spine', 'ShoulderCenter', 'Head',
    'ShoulderLeft', 'ElbowLeft', 'WristLeft', 'HandLeft',
    'ShoulderRight', 'ElbowRight', 'WristRight', 'HandRight',
    'HipLeft', 'KneeLeft', 'AnkleLeft', 'FootLeft',
    'HipRight', 'KneeRight', 'AnkleRight', 'FootRight'
]
HIP_CENTER_IDX = 0
SHOULDER_CENTER_IDX = 2
SHOULDER_LEFT_IDX = 4
SHOULDER_RIGHT_IDX = 8

# Bone connections: (child, parent) for indices 1..19
BONE_CHILD_PARENT = [
    (1, 0), (2, 0), (3, 2), (4, 2), (5, 4), (6, 5), (7, 6),
    (8, 2), (9, 8), (10, 9), (11, 10), (12, 0), (13, 12),
    (14, 13), (15, 14), (16, 0), (17, 16), (18, 17), (19, 18)
]

LEFT_RIGHT_PAIRS = [(4,8), (5,9), (6,10), (7,11), (12,16), (13,17), (14,18), (15,19)]

# ----------------------------------------------------------------------
# Feature Extraction
# ----------------------------------------------------------------------
def normalize_skeleton(coords):
    """Center at hip, align shoulders, scale by median trunk length."""
    T = coords.shape[0]
    hip = coords[:, HIP_CENTER_IDX, :]          # (T,3)
    centered = coords - hip[:, None, :]

    # Rotate so shoulder line becomes horizontal
    left = centered[:, SHOULDER_LEFT_IDX, :]
    right = centered[:, SHOULDER_RIGHT_IDX, :]
    vec = left - right                         # (T,3)
    angles = np.arctan2(vec[:, 2], vec[:, 0])  # (T,)
    cos = np.cos(-angles)[:, None]
    sin = np.sin(-angles)[:, None]
    x = centered[..., 0]
    z = centered[..., 2]
    new_x = cos * x + sin * z
    new_z = -sin * x + cos * z
    rotated = centered.copy()
    rotated[..., 0] = new_x
    rotated[..., 2] = new_z

    # Scale by median trunk length (hip to shoulder center)
    trunk = rotated[:, SHOULDER_CENTER_IDX, :]
    trunk_len = np.linalg.norm(trunk, axis=1)
    median_trunk = np.median(trunk_len)
    if median_trunk <= 0:
        median_trunk = 1.0
    scaled = rotated / median_trunk
    return scaled

def compute_skeleton_features_from_coords(coords):
    """From normalized coords (T,20,3) produce (T,351) features."""
    T = coords.shape[0]
    # Joints
    joints_flat = coords.reshape(T, 60)                     # (T,60)
    # Bone vectors
    bone_vecs = np.zeros((T, 19, 3))
    for i, (child, parent) in enumerate(BONE_CHILD_PARENT):
        bone_vecs[:, i, :] = coords[:, child, :] - coords[:, parent, :]
    bone_flat = bone_vecs.reshape(T, 57)                   # (T,57)
    base = np.concatenate([joints_flat, bone_flat], axis=1) # (T,117)

    vel = np.zeros_like(base)
    if T > 1:
        vel[1:] = base[1:] - base[:-1]
    acc = np.zeros_like(base)
    if T > 2:
        acc[1:] = vel[1:] - vel[:-1]
    return np.concatenate([base, vel, acc], axis=1).astype(np.float32)   # (T,351)

def extract_audio_features(audio_bytes, timestamps):
    """audio_bytes: wav file bytes, timestamps: (T,) in seconds -> (T,103) features."""
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)
    n_fft, hop_length, n_mels = 512, 160, 64
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft,
                                         hop_length=hop_length, n_mels=n_mels)
    logmel = np.log(mel + 1e-6)                                   # (64, t)
    mfcc = librosa.feature.mfcc(S=logmel, n_mfcc=13, sr=sr)       # (13, t)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    all_feat = np.vstack([logmel, mfcc, delta, delta2])           # (103, t)

    audio_times = librosa.frames_to_time(np.arange(all_feat.shape[1]),
                                         sr=sr, hop_length=hop_length)
    feat_resampled = np.zeros((len(timestamps), 103), dtype=np.float32)
    for i in range(103):
        interp = scipy.interpolate.interp1d(
            audio_times, all_feat[i], kind='linear',
            bounds_error=False, fill_value='extrapolate')
        feat_resampled[:, i] = interp(timestamps)
    return feat_resampled

def parse_mat(mat_bytes):
    """Robust extraction of coords, labels, frame rate from _data.mat bytes."""
    mat = scipy.io.loadmat(io.BytesIO(mat_bytes), squeeze_me=False, struct_as_record=False)
    video = mat['Video'][0, 0]  # mat_struct

    # NumFrames and FrameRate
    num_frames = int(video.NumFrames[0, 0])
    frame_rate = float(video.FrameRate[0, 0])

    # Frames: could be (1, num_frames) or (num_frames, 1)
    frames_arr = video.Frames
    frames_list = []
    if frames_arr.shape[0] == 1 and frames_arr.shape[1] == num_frames:
        frames_list = [frames_arr[0, i] for i in range(num_frames)]
    elif frames_arr.shape[1] == 1 and frames_arr.shape[0] == num_frames:
        frames_list = [frames_arr[i, 0] for i in range(num_frames)]
    else:
        # Fallback: flatten and hope order is correct
        frames_list = frames_arr.flatten()

    coords = []
    for f in frames_list:
        skel = f.Skeleton[0, 0]  # mat_struct
        world_pos = skel.WorldPosition  # (20, 3) array
        coords.append(world_pos)
    coords = np.stack(coords, axis=0)  # (T, 20, 3)

    # Labels (if present)
    labels = np.zeros(num_frames, dtype=int)
    if hasattr(video, 'Labels'):
        labels_data = video.Labels
        if labels_data.size > 0:
            # Determine shape
            if labels_data.shape[0] == 1:
                labs = [labels_data[0, i] for i in range(labels_data.shape[1])]
            elif labels_data.shape[1] == 1:
                labs = [labels_data[i, 0] for i in range(labels_data.shape[0])]
            else:
                labs = labels_data.flatten()
            for lab in labs:
                # Extract gesture name
                name_field = lab.Name
                name = ''
                if isinstance(name_field, np.ndarray):
                    if name_field.dtype.kind == 'U':
                        name = ''.join(name_field.ravel())
                    else:
                        # assume ascii bytes
                        name = ''.join(chr(x) for x in name_field.ravel())
                elif isinstance(name_field, bytes):
                    name = name_field.decode()
                else:
                    name = str(name_field)
                name = name.strip()
                label_id = GESTURE_TO_ID.get(name, 0)
                if label_id > 0:
                    begin = int(lab.Begin[0, 0])
                    end = int(lab.End[0, 0])
                    labels[begin:end+1] = label_id
    return coords, labels, frame_rate

def process_sample(mat_bytes, audio_bytes):
    """Main processing: returns (coords_norm, audio_feat, labels)."""
    coords, labels, frame_rate = parse_mat(mat_bytes)
    coords_norm = normalize_skeleton(coords)
    timestamps = np.arange(coords_norm.shape[0]) / frame_rate
    audio_feat = extract_audio_features(audio_bytes, timestamps)
    return coords_norm, audio_feat, labels

# ----------------------------------------------------------------------
# Data Loading (from tar.gz archives)
# ----------------------------------------------------------------------
INPUT_DIR = './input'
TRAINING_CSV = os.path.join(INPUT_DIR, 'training.csv')
TEST_CSV = os.path.join(INPUT_DIR, 'test.csv')
SUBMISSION_FILE = './submission/submission.csv'
WORKING_DIR = './working'
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(os.path.dirname(SUBMISSION_FILE), exist_ok=True)

CACHE_PATH = os.path.join(WORKING_DIR, 'features_cache.pkl')

def load_samples(id_set, archive_paths):
    """Return dict {sample_id: (coords, audio, labels)}."""
    data = {}
    for arch_path in archive_paths:
        if not os.path.exists(arch_path):
            continue
        with tarfile.open(arch_path, 'r:gz') as tar:
            members = [m for m in tar.getmembers() if m.name.endswith('.zip')]
            for member in tqdm(members, desc=f"Reading {os.path.basename(arch_path)}"):
                base = os.path.basename(member.name)
                try:
                    sample_id = int(base[6:11])       # Sample00001.zip -> 1
                except:
                    continue
                if sample_id in id_set and sample_id not in data:
                    f = tar.extractfile(member)
                    zip_bytes = f.read()
                    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
                        mat_bytes, audio_bytes = None, None
                        for name in z.namelist():
                            if name.endswith('_data.mat'):
                                with z.open(name) as fmat:
                                    mat_bytes = fmat.read()
                            elif name.endswith('_audio.wav'):
                                with z.open(name) as fau:
                                    audio_bytes = fau.read()
                        if mat_bytes is None or audio_bytes is None:
                            continue
                        try:
                            coords, audio, labels = process_sample(mat_bytes, audio_bytes)
                            data[sample_id] = (coords, audio, labels)
                        except Exception as e:
                            print(f"Error processing {sample_id}: {e}")
        if len(data) == len(id_set):
            break
    return data

def load_all_data(train_ids, test_ids):
    """Load (or read from cache) training and test data."""
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, 'rb') as f:
            train_data, test_data = pickle.load(f)
        if set(train_data.keys()) == set(train_ids) and set(test_data.keys()) == set(test_ids):
            return train_data, test_data

    # Locate archives
    all_files = os.listdir(INPUT_DIR)
    train_archives = [os.path.join(INPUT_DIR, f) for f in all_files
                      if f.startswith('training') and f.endswith('.tar.gz')]
    test_archives = [os.path.join(INPUT_DIR, f) for f in all_files
                     if f.startswith('test') and f.endswith('.tar.gz')]

    train_data = load_samples(set(train_ids), train_archives)
    test_data = load_samples(set(test_ids), test_archives)

    # Cache
    with open(CACHE_PATH, 'wb') as f:
        pickle.dump((train_data, test_data), f)
    return train_data, test_data

# ----------------------------------------------------------------------
# Dataset with on‑the‑fly augmentation & feature computation
# ----------------------------------------------------------------------
class GestureDataset(Dataset):
    def __init__(self, samples, augment=False):
        self.samples = samples          # list of (coords, audio, labels)
        self.augment = augment
        self.left_right_pairs = LEFT_RIGHT_PAIRS

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        coords, audio, labels = self.samples[idx]
        if self.augment:
            coords, audio, labels = self._augment(coords, audio, labels)
        # Compute skeleton features from current (possibly augmented) coords
        skel_feat = self._compute_skeleton_features(coords)   # (T,351)
        features = np.concatenate([skel_feat, audio], axis=1) # (T,454)
        return (torch.from_numpy(features).float(),
                torch.from_numpy(labels).long())

    def _compute_skeleton_features(self, coords):
        return compute_skeleton_features_from_coords(coords)

    def _augment(self, coords, audio, labels):
        # Time scaling
        if np.random.rand() < 0.5:
            scale = np.random.uniform(0.8, 1.2)
            T = coords.shape[0]
            new_T = max(1, int(T * scale))
            x_old = np.linspace(0, 1, T)
            x_new = np.linspace(0, 1, new_T)
            # Coords
            coords_new = np.zeros((new_T, 20, 3))
            for j in range(20):
                for d in range(3):
                    interp = scipy.interpolate.interp1d(x_old, coords[:, j, d], kind='linear', assume_sorted=True)
                    coords_new[:, j, d] = interp(x_new)
            # Audio
            audio_new = np.zeros((new_T, audio.shape[1]))
            for d in range(audio.shape[1]):
                interp = scipy.interpolate.interp1d(x_old, audio[:, d], kind='linear', assume_sorted=True)
                audio_new[:, d] = interp(x_new)
            # Labels (nearest)
            idx = np.round(x_new * (T-1)).astype(int)
            labels_new = labels[idx]
            coords, audio, labels = coords_new, audio_new, labels_new

        # Mirror
        if np.random.rand() < 0.5:
            coords_mir = coords.copy()
            coords_mir[..., 0] = -coords_mir[..., 0]
            for l, r in self.left_right_pairs:
                coords_mir[:, [l, r], :] = coords_mir[:, [r, l], :]
            coords = coords_mir

        # Rotation around Y
        if np.random.rand() < 0.5:
            angle = np.random.uniform(-10, 10) * np.pi / 180
            cos, sin = np.cos(angle), np.sin(angle)
            x, z = coords[..., 0].copy(), coords[..., 2].copy()
            coords[..., 0] = x * cos + z * sin
            coords[..., 2] = -x * sin + z * cos

        # Add noise
        if np.random.rand() < 0.5:
            noise = np.random.normal(0, 0.001, coords.shape)
            coords += noise

        # Audio gain
        if np.random.rand() < 0.3:
            audio *= np.random.uniform(0.9, 1.1)

        return coords, audio, labels

def collate_fn(batch):
    """Pad variable length sequences, sort descending."""
    batch.sort(key=lambda x: x[0].shape[0], reverse=True)
    features, labels = zip(*batch)
    lengths = [f.shape[0] for f in features]
    max_len = max(lengths)
    feat_dim = features[0].shape[1]
    padded_features = torch.zeros((len(batch), max_len, feat_dim), dtype=torch.float32)
    padded_labels = torch.full((len(batch), max_len), fill_value=-1, dtype=torch.long)
    for i, (f, l) in enumerate(zip(features, labels)):
        padded_features[i, :lengths[i], :] = f
        padded_labels[i, :lengths[i]] = l
    return padded_features, padded_labels, torch.tensor(lengths, dtype=torch.long)

# ----------------------------------------------------------------------
# Neural Network
# ----------------------------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=kernel_size//2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(out_channels, out_channels//16, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels//16, out_channels, 1),
            nn.Sigmoid()
        )
        self.downsample = None
        if in_channels != out_channels or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        se_weight = self.se(out)
        out = out * se_weight
        out = self.dropout(out)

        if self.downsample is not None:
            identity = self.downsample(identity)
        out += identity
        out = self.relu(out)
        return out

class MultiModalGestureModel(nn.Module):
    def __init__(self, skeleton_in=351, audio_in=103):
        super().__init__()
        # Skeleton branch
        self.skel_conv1 = nn.Conv1d(skeleton_in, 128, kernel_size=5, padding=2, bias=False)
        self.skel_bn1 = nn.BatchNorm1d(128)
        self.skel_relu = nn.ReLU(inplace=True)
        self.skel_block1 = ResidualBlock(128, 128)
        self.skel_block2 = ResidualBlock(128, 256)
        self.skel_block3 = ResidualBlock(256, 384)

        # Audio branch
        self.audio_conv1 = nn.Conv1d(audio_in, 64, kernel_size=5, padding=2, bias=False)
        self.audio_bn1 = nn.BatchNorm1d(64)
        self.audio_relu = nn.ReLU(inplace=True)
        self.audio_conv2 = nn.Conv1d(64, 384, kernel_size=5, padding=2, bias=False)
        self.audio_bn2 = nn.BatchNorm1d(384)

        # Fusion
        self.lstm = nn.LSTM(input_size=384+384, hidden_size=384, num_layers=2,
                            bidirectional=True, dropout=0.5, batch_first=True)
        self.classifier = nn.Linear(384*2, 21)   # 0..20
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, lengths):
        # x: (B, T, 454)
        skel = x[..., :351].permute(0, 2, 1)   # (B,351,T)
        audio = x[..., 351:].permute(0, 2, 1)  # (B,103,T)

        # Skeleton
        skel = self.skel_conv1(skel)
        skel = self.skel_bn1(skel)
        skel = self.skel_relu(skel)
        skel = self.skel_block1(skel)
        skel = self.skel_block2(skel)
        skel = self.skel_block3(skel)          # (B,384,T)

        # Audio
        audio = self.audio_conv1(audio)
        audio = self.audio_bn1(audio)
        audio = self.audio_relu(audio)
        audio = self.audio_conv2(audio)
        audio = self.audio_bn2(audio)
        audio = self.audio_relu(audio)         # (B,384,T)

        fused = torch.cat([skel, audio], dim=1).permute(0, 2, 1)  # (B,T,768)

        packed = nn.utils.rnn.pack_padded_sequence(fused, lengths.cpu(),
                                                   batch_first=True, enforce_sorted=True)
        packed_out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)  # (B,T,768)
        out = self.dropout(out)
        logits = self.classifier(out)          # (B,T,21)
        return logits

# ----------------------------------------------------------------------
# Utility: Levenshtein, Viterbi, Post‑processing
# ----------------------------------------------------------------------
def levenshtein_distance(a, b):
    if len(a) < len(b):
        return levenshtein_distance(b, a)
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            ins = prev[j + 1] + 1
            dele = curr[j] + 1
            sub = prev[j] + (ca != cb)
            curr.append(min(ins, dele, sub))
        prev = curr
    return prev[-1]

def postprocess(path):
    """Remove non‑gesture (0) and collapse consecutive duplicates."""
    seq = []
    prev = None
    for lab in path:
        if lab != 0:
            if lab != prev:
                seq.append(lab)
            prev = lab
    return seq

def compute_transition_probs(samples):
    """From list of (_, _, labels) return init_log and trans_log."""
    eps = 1e-6
    trans = np.ones((21, 21)) * eps
    init = np.ones(21) * eps
    for _, _, labels in samples:
        if len(labels) == 0:
            continue
        init[labels[0]] += 1
        for i in range(len(labels)-1):
            trans[labels[i], labels[i+1]] += 1
    init = init / init.sum()
    trans = trans / trans.sum(axis=1, keepdims=True)
    return np.log(init + 1e-12), np.log(trans + 1e-12)

def viterbi_decode(log_probs, trans_log, init_log):
    T, N = log_probs.shape
    v = np.full((T, N), -np.inf)
    back = np.zeros((T, N), dtype=int)
    v[0] = init_log + log_probs[0]
    for t in range(1, T):
        for j in range(N):
            trans = v[t-1] + trans_log[:, j]
            best = np.max(trans)
            back[t, j] = np.argmax(trans)
            v[t, j] = best + log_probs[t, j]
    path = np.zeros(T, dtype=int)
    path[-1] = np.argmax(v[-1])
    for t in range(T-2, -1, -1):
        path[t] = back[t+1, path[t+1]]
    return path.tolist()

# ----------------------------------------------------------------------
# Evaluation & Prediction helpers
# ----------------------------------------------------------------------
def compute_features_for_inference(coords, audio):
    """Combined features without augmentation."""
    skel = compute_skeleton_features_from_coords(coords)   # (T,351)
    return np.concatenate([skel, audio], axis=1)          # (T,454)

def mirror_coords(coords):
    coords_mir = coords.copy()
    coords_mir[..., 0] = -coords_mir[..., 0]
    for l, r in LEFT_RIGHT_PAIRS:
        coords_mir[:, [l, r], :] = coords_mir[:, [r, l], :]
    return coords_mir

def predict_sample(model, coords, audio, device, use_mirror=True):
    model.eval()
    with torch.no_grad():
        feats = compute_features_for_inference(coords, audio)
        logits = model(torch.from_numpy(feats).unsqueeze(0).float().to(device),
                       torch.tensor([feats.shape[0]]).to(device))
        probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()   # (T,21)
        if use_mirror:
            coords_mir = mirror_coords(coords)
            feats_mir = compute_features_for_inference(coords_mir, audio)
            logits_mir = model(torch.from_numpy(feats_mir).unsqueeze(0).float().to(device),
                               torch.tensor([feats_mir.shape[0]]).to(device))
            probs_mir = F.softmax(logits_mir, dim=-1).squeeze(0).cpu().numpy()
            probs = (probs + probs_mir) / 2
        return probs

def evaluate(model, samples, device, init_log, trans_log):
    """Return Levenshtein error rate on given samples."""
    distances = []
    total_gestures = 0
    for coords, audio, labels in samples:
        true_seq = postprocess(labels)
        probs = predict_sample(model, coords, audio, device, use_mirror=False)
        log_probs = np.log(probs + 1e-12)
        path = viterbi_decode(log_probs, trans_log, init_log)
        pred_seq = postprocess(path)
        dist = levenshtein_distance(pred_seq, true_seq)
        distances.append(dist)
        total_gestures += len(true_seq)
    total_dist = sum(distances)
    return total_dist / total_gestures if total_gestures > 0 else float('inf'), total_dist, total_gestures

# ----------------------------------------------------------------------
# Training Loop
# ----------------------------------------------------------------------
def train_fold(train_samples, val_samples, init_log, trans_log,
               epochs=50, batch_size=8, lr=0.001, fold_idx=0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset = GestureDataset(train_samples, augment=True)
    val_dataset = GestureDataset(val_samples, augment=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=4, pin_memory=True)
    # Model
    model = MultiModalGestureModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=0.05)

    best_val_err = float('inf')
    for epoch in range(1, epochs+1):
        model.train()
        pbar = tqdm(train_loader, desc=f'Fold {fold_idx} Epoch {epoch}')
        total_loss = 0.0
        for feats, labs, lens in pbar:
            feats, labs = feats.to(device), labs.to(device)
            optimizer.zero_grad()
            logits = model(feats, lens.to(device))
            B, T, C = logits.shape

            # CE loss
            loss_ce = criterion(logits.view(-1, C), labs.view(-1))

            # Smoothness loss on valid frames only
            probs = F.softmax(logits, dim=-1)
            mask = torch.zeros((B, T), device=device)
            for i, l in enumerate(lens):
                mask[i, :l] = 1
            mask_pair = mask[:, 1:] * mask[:, :-1]
            diff = probs[:, 1:, :] - probs[:, :-1, :]
            smooth = (diff ** 2).sum(dim=2) * mask_pair
            smooth_loss = smooth.sum() / (mask_pair.sum() + 1e-8)

            loss = loss_ce + 0.2 * smooth_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        scheduler.step()
        # Validate
        val_err, _, _ = evaluate(model, val_samples, device, init_log, trans_log)
        tqdm.write(f'Fold {fold_idx} Epoch {epoch} – Val error: {val_err:.4f}')
        if val_err < best_val_err:
            best_val_err = val_err
            torch.save(model.state_dict(), os.path.join(WORKING_DIR, f'best_fold{fold_idx}.pt'))

    # Load best model for this fold
    model.load_state_dict(torch.load(os.path.join(WORKING_DIR, f'best_fold{fold_idx}.pt')))
    return model, best_val_err

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
if __name__ == '__main__':
    # 1. Load IDs
    train_df = pd.read_csv(TRAINING_CSV)
    train_ids = train_df['Id'].tolist()
    test_df = pd.read_csv(TEST_CSV)
    test_ids = test_df['Id'].tolist()

    # 2. Load data
    print("Loading data...")
    train_data, test_data = load_all_data(train_ids, test_ids)

    # 3. Build sample list for training (only those we actually have)
    train_samples = []
    missing = []
    for sid in train_ids:
        if sid in train_data:
            train_samples.append(train_data[sid])
        else:
            missing.append(sid)
    if missing:
        print(f"Warning: missing training samples {missing} – they will be ignored.")

    # 4. 5‑fold cross‑validation (per‑fold transition probabilities)
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    indices = list(range(len(train_samples)))
    fold_errors = []
    fold_models = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
        print(f"\n===== Fold {fold+1}/{n_splits} =====")
        fold_train = [train_samples[i] for i in train_idx]
        fold_val = [train_samples[i] for i in val_idx]

        # Compute transition probabilities from this fold's training data
        init_log, trans_log = compute_transition_probs(fold_train)

        model, val_err = train_fold(fold_train, fold_val, init_log, trans_log,
                                    epochs=50, batch_size=8, fold_idx=fold+1)
        fold_errors.append(val_err)
        fold_models.append(model)

    avg_error = np.mean(fold_errors)
    print(f"\nCross‑validation Levenshtein error: {avg_error:.4f} (avg over {n_splits} folds)")

    # 5. Train final model on all training data
    print("\nTraining final model on all training data...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    full_dataset = GestureDataset(train_samples, augment=True)
    full_loader = DataLoader(full_dataset, batch_size=8, shuffle=True,
                             collate_fn=collate_fn, num_workers=4, pin_memory=True)
    final_model = MultiModalGestureModel().to(device)
    optimizer = optim.Adam(final_model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    criterion = nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=0.05)

    for epoch in range(1, 51):
        final_model.train()
        pbar = tqdm(full_loader, desc=f'Final Epoch {epoch}')
        for feats, labs, lens in pbar:
            feats, labs = feats.to(device), labs.to(device)
            optimizer.zero_grad()
            logits = final_model(feats, lens.to(device))
            B, T, C = logits.shape
            loss_ce = criterion(logits.view(-1, C), labs.view(-1))
            probs = F.softmax(logits, dim=-1)
            mask = torch.zeros((B, T), device=device)
            for i, l in enumerate(lens):
                mask[i, :l] = 1
            mask_pair = mask[:, 1:] * mask[:, :-1]
            diff = probs[:, 1:, :] - probs[:, :-1, :]
            smooth = (diff ** 2).sum(dim=2) * mask_pair
            smooth_loss = smooth.sum() / (mask_pair.sum() + 1e-8)
            loss = loss_ce + 0.2 * smooth_loss
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=loss.item())
        scheduler.step()

    # 6. Compute transition probabilities from all training data for decoding
    init_log, trans_log = compute_transition_probs(train_samples)

    # 7. Predict test set and write submission
    print("Predicting test set...")
    test_predictions = {}
    for sid in test_ids:
        if sid not in test_data:
            # Fallback: predict empty sequence
            test_predictions[sid] = []
            continue
        coords, audio, _ = test_data[sid]
        probs = predict_sample(final_model, coords, audio, device, use_mirror=True)
        log_probs = np.log(probs + 1e-12)
        path = viterbi_decode(log_probs, trans_log, init_log)
        gest_seq = postprocess(path)
        test_predictions[sid] = gest_seq

    # Write submission.csv
    with open(SUBMISSION_FILE, 'w') as f:
        f.write("Id,Sequence\n")
        for sid in sorted(test_ids):
            seq_str = ' '.join(str(g) for g in test_predictions[sid])
            f.write(f"{sid},{seq_str}\n")
    print(f"Submission saved to {SUBMISSION_FILE}")