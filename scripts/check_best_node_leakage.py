#!/usr/bin/env python3
"""
MLE-Lite best-node leakage checker.

This script is designed for the MLE-Lite competition set.

It performs:
1. Exact sample-level hashing
   - Every train/val/test/external sample gets its own hash record
   - Every test sample has an explicit per-sample hash
   - Reports exact overlap between train-like and test-like samples

2. Modality-aware near-duplicate analysis
   - Image: pHash + Hamming similarity
   - Text / Seq2Seq: TF-IDF + nearest neighbor cosine similarity
   - Tabular: row serialization + TF-IDF nearest neighbor
   - Audio: librosa MFCC if available, otherwise byte sketch fallback

3. Best-node code scan
   - Detects suspicious code patterns such as train/test joins or all-data fitting

Usage:
python scripts/check_mle_lite_leakage.py \
  --run-dir <run_dir> \
  --best-node-id <best_node_id>

Optional:
python scripts/check_mle_lite_leakage.py \
  --run-dir <run_dir> \
  --best-node-id <best_node_id> \
  --nn-threshold 0.95 \
  --max-nn-train 5000 \
  --max-nn-test 2000
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    import yaml
except ImportError:
    yaml = None

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    from sklearn.neighbors import NearestNeighbors
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError:
    NearestNeighbors = None
    TfidfVectorizer = None

try:
    import librosa
except Exception:
    librosa = None


# ============================================================
# MLE-Lite competition category mapping
# ============================================================

MLE_LITE_CATEGORIES = {
    "aerial-cactus-identification": "image_classification",
    "aptos2019-blindness-detection": "image_classification",
    "new-york-city-taxi-fare-prediction": "tabular",
    "plant-pathology-2020-fgvc7": "image_classification",
    "ranzcr-clip-catheter-line-classification": "image_classification",
    "spooky-author-identification": "text_classification",
    "mlsp-2013-birds": "audio_classification",
    "detecting-insults-in-social-commentary": "text_classification",
    "dog-breed-identification": "image_classification",
    "dogs-vs-cats-redux-kernels-edition": "image_classification",
    "histopathologic-cancer-detection": "image_regression",
    "jigsaw-toxic-comment-classification-challenge": "text_classification",
    "nomad2018-predict-transparent-conductors": "tabular",
    "random-acts-of-pizza": "text_classification",
    "text-normalization-challenge-english-language": "seq2seq",
    "denoising-dirty-documents": "image_to_image",
    "leaf-classification": "image_classification",
    "siim-isic-melanoma-classification": "image_classification",
    "tabular-playground-series-dec-2021": "tabular",
    "tabular-playground-series-may-2022": "tabular",
    "tabular-playground-series-dec-2021-v2": "tabular",
    "tabular-playground-series-may-2022-v2": "tabular",
    "text-normalization-challenge-russian-language": "seq2seq",
    "the-icml-2013-whale-challenge-right-whale-redux": "audio_classification",
}


IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"
}

AUDIO_EXTS = {
    ".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".aif", ".aiff"
}

TEXT_EXTS = {
    ".txt", ".jsonl", ".json", ".md"
}

TABLE_EXTS = {
    ".csv", ".tsv"
}

SUSPICIOUS_PATTERNS = [
    r"concat\s*\(",
    r"merge\s*\(",
    r"join\s*\(",
    r"fit\s*\(\s*all",
    r"train.*test",
    r"test.*train",
    r"sample_submission",
    r"pseudo.?label",
    r"all_data",
    r"full_data",
]


# ============================================================
# Data classes
# ============================================================

@dataclass
class SampleRecord:
    split: str
    sample_id: str
    source_path: str
    hash: str
    hash_type: str
    modality: str
    extra: dict[str, Any]


# ============================================================
# Basic utilities
# ============================================================

def safe_read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def hash_text(text: str) -> str:
    return sha256_bytes(normalize_text(text).encode("utf-8"))


def hash_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def ratio(num: int, den: int) -> float:
    return 0.0 if den == 0 else num / den


def deterministic_sample(seq: list[Any], k: int) -> list[Any]:
    if len(seq) <= k:
        return seq
    rng = random.Random(42)
    idxs = list(range(len(seq)))
    rng.shuffle(idxs)
    idxs = sorted(idxs[:k])
    return [seq[i] for i in idxs]


def l2_normalize(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return arr / norms


# ============================================================
# Config / run helpers
# ============================================================

def find_run_config(run_dir: Path) -> Path:
    config_path = run_dir / "config.yaml"
    if config_path.exists():
        return config_path
    raise FileNotFoundError(f"Cannot find config.yaml under {run_dir}")


def parse_config(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("Missing dependency: pyyaml. Install with: pip install pyyaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def infer_dataset_root(cfg: dict[str, Any]) -> tuple[str, Path]:
    data_root = cfg.get("data_root")
    exp_id = cfg.get("exp_id")

    if not data_root or not exp_id:
        raise ValueError("config.yaml must contain data_root and exp_id")

    return str(exp_id), Path(data_root) / str(exp_id)


def find_best_code_path(run_dir: Path, node_id: str) -> Path:
    matches = list(run_dir.rglob(f"code_{node_id}.py"))
    if matches:
        return matches[0]

    matches = list(run_dir.rglob(f"code_{node_id}_template.py"))
    if matches:
        return matches[0]

    raise FileNotFoundError(f"Cannot find best-node code file for node_id={node_id}")


# ============================================================
# Modality / category helpers
# ============================================================

def category_to_modality(category: str) -> str:
    if category in {"image_classification", "image_regression", "image_to_image"}:
        return "vision"
    if category == "audio_classification":
        return "audio"
    if category in {"text_classification", "seq2seq"}:
        return "text"
    if category == "tabular":
        return "tabular"
    return "unknown"


def detect_code_mode(code_text: str) -> str:
    image_signals = [
        r"Image\.open",
        r"cv2\.imread",
        r"torchvision",
        r"PIL",
        r"\.jpg",
        r"\.jpeg",
        r"\.png",
    ]

    audio_signals = [
        r"\.wav",
        r"\.mp3",
        r"\.flac",
        r"\.ogg",
        r"\.aif",
        r"\.aiff",
        r"librosa",
        r"torchaudio",
        r"soundfile",
    ]

    csv_signals = [
        r"read_csv",
        r"train\.csv",
        r"test\.csv",
        r"val\.csv",
        r"validation\.csv",
    ]

    text_signals = [
        r"tokenizer",
        r"AutoTokenizer",
        r"BertTokenizer",
        r"T5Tokenizer",
        r"GPT2Tokenizer",
        r"RobertaTokenizer",
        r"datasets\.load_dataset",
        r"load_dataset\(",
        r"\.txt",
        r"\.jsonl",
        r"\btext\b",
        r"\bsource\b",
        r"\btarget\b",
        r"\binput_text\b",
        r"\boutput_text\b",
        r"\bprompt\b",
        r"\bresponse\b",
    ]

    has_image = any(re.search(p, code_text, flags=re.IGNORECASE) for p in image_signals)
    has_audio = any(re.search(p, code_text, flags=re.IGNORECASE) for p in audio_signals)
    has_csv = any(re.search(p, code_text, flags=re.IGNORECASE) for p in csv_signals)
    has_text = any(re.search(p, code_text, flags=re.IGNORECASE) for p in text_signals)

    if has_image and has_csv:
        return "vision_via_manifest"
    if has_audio and has_csv:
        return "audio_via_manifest"
    if has_text and has_csv:
        return "text_via_manifest"

    if has_image:
        return "vision"
    if has_audio:
        return "audio"
    if has_text:
        return "text"
    if has_csv:
        return "tabular"
    return "unknown"


def decide_effective_mode(category_modality: str, code_mode: str) -> str:
    """
    Prefer explicit best-node code behavior when it gives manifest mode.
    Otherwise use known MLE-Lite category.
    """
    if code_mode in {"vision_via_manifest", "audio_via_manifest", "text_via_manifest"}:
        return code_mode

    if category_modality == "vision":
        return "vision"
    if category_modality == "audio":
        return "audio"
    if category_modality == "text":
        return "text"
    if category_modality == "tabular":
        return "tabular"

    return code_mode


# ============================================================
# Split discovery
# ============================================================

def infer_split_paths(dataset_root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}

    prepared = dataset_root / "prepared"
    public = prepared / "public"
    private = prepared / "private"

    file_candidates = [
        ("train", public / "train.csv"),
        ("val", public / "val.csv"),
        ("val", public / "valid.csv"),
        ("val", public / "validation.csv"),
        ("test", public / "test.csv"),
        ("test", public / "test2.csv"),

        ("train", public / "train.tsv"),
        ("val", public / "val.tsv"),
        ("test", public / "test.tsv"),

        ("train", public / "train.jsonl"),
        ("val", public / "val.jsonl"),
        ("test", public / "test.jsonl"),

        ("train", public / "train.txt"),
        ("val", public / "val.txt"),
        ("test", public / "test.txt"),

        ("private_test_meta", private / "test.csv"),
        ("private_test_meta", private / "test.tsv"),
        ("private_test_meta", private / "test.jsonl"),

        ("train", dataset_root / "train.csv"),
        ("val", dataset_root / "val.csv"),
        ("test", dataset_root / "test.csv"),
    ]

    for split, path in file_candidates:
        if path.exists() and split not in result:
            result[split] = path

    dir_candidates = {
        "train": [
            public / "train",
            public / "training",
            prepared / "train",
            dataset_root / "train",
        ],
        "val": [
            public / "val",
            public / "valid",
            public / "validation",
            prepared / "val",
            dataset_root / "val",
        ],
        "test": [
            public / "test",
            public / "test2",
            public / "testing",
            public / "public_test",
            prepared / "test",
            dataset_root / "test",
        ],
        "external": [
            public / "external",
            prepared / "external",
            dataset_root / "external",
        ],
    }

    for split, paths in dir_candidates.items():
        if split in result:
            continue
        for p in paths:
            if p.exists():
                result[split] = p
                break

    return result


def find_asset_roots(dataset_root: Path, exts: set[str]) -> list[Path]:
    roots = [
        dataset_root,
        dataset_root / "prepared",
        dataset_root / "prepared" / "public",
        dataset_root / "prepared" / "private",
    ]

    found: list[Path] = []
    seen: set[str] = set()

    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            try:
                if p.is_file() and p.suffix.lower() in exts:
                    parent = p.parent
                    if str(parent) not in seen:
                        found.append(parent)
                        seen.add(str(parent))
                elif p.is_dir():
                    has_target = any(
                        child.is_file() and child.suffix.lower() in exts
                        for child in p.iterdir()
                    )
                    if has_target and str(p) not in seen:
                        found.append(p)
                        seen.add(str(p))
            except Exception:
                continue

    return found


def build_asset_index(asset_roots: list[Path], exts: set[str]) -> dict[str, Path]:
    """
    Map file stem -> file path.
    Example:
      images/123.jpg -> {"123": images/123.jpg}
    """
    index: dict[str, Path] = {}
    for root in asset_roots:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                index.setdefault(p.stem, p)
    return index


# ============================================================
# Manifest / tabular / text reading
# ============================================================

def delimiter_for(path: Path) -> str:
    return "\t" if path.suffix.lower() == ".tsv" else ","


def read_manifest_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or not path.is_file():
        return []

    if path.suffix.lower() not in {".csv", ".tsv"}:
        return []

    rows: list[dict[str, str]] = []
    with open(path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter_for(path))
        if reader.fieldnames is None:
            return []
        for i, row in enumerate(reader):
            normalized = {
                str(k): "" if v is None else str(v).strip()
                for k, v in row.items()
            }
            normalized["_row_idx"] = str(i)
            rows.append(normalized)

    return rows


def row_sample_id(row: dict[str, str], fallback_prefix: str) -> str:
    for key in ["id", "Id", "ID", "image_id", "ImageId", "filename", "file", "path"]:
        if key in row and row[key]:
            return row[key]
    return f"{fallback_prefix}_row_{row.get('_row_idx', 'unknown')}"


def hash_row(row: dict[str, str]) -> str:
    clean = {k: v for k, v in row.items() if k != "_row_idx"}
    payload = json.dumps(clean, sort_keys=True, ensure_ascii=False)
    return sha256_bytes(payload.encode("utf-8"))


def read_jsonl_records(path: Path, split: str) -> list[SampleRecord]:
    records: list[SampleRecord] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            sample_id = f"{split}_jsonl_{i}"
            try:
                obj = json.loads(line)
                payload = json.dumps(obj, sort_keys=True, ensure_ascii=False)
            except Exception:
                payload = line
            records.append(
                SampleRecord(
                    split=split,
                    sample_id=sample_id,
                    source_path=str(path),
                    hash=sha256_bytes(payload.encode("utf-8")),
                    hash_type="sha256_jsonl_record",
                    modality="text",
                    extra={"row_idx": i},
                )
            )
    return records


def read_txt_records(path: Path, split: str) -> list[SampleRecord]:
    records: list[SampleRecord] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            records.append(
                SampleRecord(
                    split=split,
                    sample_id=f"{split}_line_{i}",
                    source_path=str(path),
                    hash=hash_text(line),
                    hash_type="sha256_normalized_text_line",
                    modality="text",
                    extra={"line_idx": i},
                )
            )
    return records


# ============================================================
# Exact hash record builders
# ============================================================

def collect_file_records_from_dir(
    path: Path,
    split: str,
    exts: set[str],
    modality: str,
    hash_type: str = "sha256_file",
) -> list[SampleRecord]:
    records: list[SampleRecord] = []

    if not path.exists():
        return records

    files: list[Path] = []
    if path.is_file():
        if path.suffix.lower() in exts:
            files = [path]
    else:
        files = sorted(
            p for p in path.rglob("*")
            if p.is_file() and p.suffix.lower() in exts
        )

    for p in files:
        try:
            records.append(
                SampleRecord(
                    split=split,
                    sample_id=p.stem,
                    source_path=str(p),
                    hash=hash_file(p),
                    hash_type=hash_type,
                    modality=modality,
                    extra={"suffix": p.suffix.lower()},
                )
            )
        except Exception as exc:
            records.append(
                SampleRecord(
                    split=split,
                    sample_id=p.stem,
                    source_path=str(p),
                    hash="",
                    hash_type="error",
                    modality=modality,
                    extra={"error": str(exc)},
                )
            )

    return records


def collect_table_records(path: Path, split: str, modality: str = "tabular") -> list[SampleRecord]:
    records: list[SampleRecord] = []
    rows = read_manifest_rows(path)

    for row in rows:
        sid = row_sample_id(row, split)
        records.append(
            SampleRecord(
                split=split,
                sample_id=sid,
                source_path=str(path),
                hash=hash_row(row),
                hash_type="sha256_row_json",
                modality=modality,
                extra={"row_idx": row.get("_row_idx")},
            )
        )

    return records


def collect_text_records(path: Path, split: str) -> list[SampleRecord]:
    if path.suffix.lower() in {".csv", ".tsv"}:
        return collect_table_records(path, split, modality="text_manifest")
    if path.suffix.lower() == ".jsonl":
        return read_jsonl_records(path, split)
    if path.suffix.lower() in {".txt", ".md", ".json"}:
        return read_txt_records(path, split)
    return []


def collect_manifest_asset_records(
    manifest_path: Path,
    split: str,
    asset_index: dict[str, Path],
    modality: str,
) -> list[SampleRecord]:
    """
    For manifest-based image/audio tasks:
    train.csv/test.csv contains IDs; actual samples are assets like <id>.jpg / <id>.aif.
    This function hashes the actual asset file, not only the CSV row.
    """
    records: list[SampleRecord] = []
    rows = read_manifest_rows(manifest_path)

    for row in rows:
        sid = row_sample_id(row, split)
        asset_path = asset_index.get(Path(sid).stem) or asset_index.get(sid)

        if asset_path is None:
            records.append(
                SampleRecord(
                    split=split,
                    sample_id=sid,
                    source_path=str(manifest_path),
                    hash="",
                    hash_type="missing_asset",
                    modality=modality,
                    extra={"row_idx": row.get("_row_idx")},
                )
            )
            continue

        try:
            records.append(
                SampleRecord(
                    split=split,
                    sample_id=sid,
                    source_path=str(asset_path),
                    hash=hash_file(asset_path),
                    hash_type="sha256_asset_file_via_manifest",
                    modality=modality,
                    extra={
                        "manifest_path": str(manifest_path),
                        "row_idx": row.get("_row_idx"),
                    },
                )
            )
        except Exception as exc:
            records.append(
                SampleRecord(
                    split=split,
                    sample_id=sid,
                    source_path=str(asset_path),
                    hash="",
                    hash_type="error",
                    modality=modality,
                    extra={"error": str(exc)},
                )
            )

    return records


def build_sample_records(
    effective_mode: str,
    split_paths: dict[str, Path],
    image_asset_roots: list[Path],
    audio_asset_roots: list[Path],
) -> dict[str, list[SampleRecord]]:
    """
    Build per-sample hash records for each split.
    Every test sample gets its own SampleRecord.
    """
    records = {
        "train": [],
        "val": [],
        "test": [],
        "external": [],
        "private_test_meta": [],
    }

    if effective_mode == "vision_via_manifest":
        asset_index = build_asset_index(image_asset_roots, IMAGE_EXTS)
        for split in ["train", "val", "test"]:
            p = split_paths.get(split)
            if p is not None:
                records[split] = collect_manifest_asset_records(
                    p, split, asset_index, modality="vision"
                )

    elif effective_mode == "audio_via_manifest":
        asset_index = build_asset_index(audio_asset_roots, AUDIO_EXTS)
        for split in ["train", "val", "test"]:
            p = split_paths.get(split)
            if p is not None:
                records[split] = collect_manifest_asset_records(
                    p, split, asset_index, modality="audio"
                )

    elif effective_mode == "vision":
        for split in ["train", "val", "test", "external"]:
            p = split_paths.get(split)
            if p is not None:
                records[split] = collect_file_records_from_dir(
                    p, split, IMAGE_EXTS, modality="vision"
                )

    elif effective_mode == "audio":
        for split in ["train", "val", "test", "external"]:
            p = split_paths.get(split)
            if p is not None:
                records[split] = collect_file_records_from_dir(
                    p, split, AUDIO_EXTS, modality="audio"
                )

    elif effective_mode == "tabular":
        for split in ["train", "val", "test", "external"]:
            p = split_paths.get(split)
            if p is not None and p.is_file() and p.suffix.lower() in {".csv", ".tsv"}:
                records[split] = collect_table_records(p, split, modality="tabular")

    elif effective_mode in {"text", "text_via_manifest"}:
        for split in ["train", "val", "test", "external"]:
            p = split_paths.get(split)
            if p is not None and p.is_file():
                records[split] = collect_text_records(p, split)

    private_meta = split_paths.get("private_test_meta")
    if private_meta is not None and private_meta.exists():
        if private_meta.suffix.lower() in {".csv", ".tsv"}:
            records["private_test_meta"] = collect_table_records(
                private_meta, "private_test_meta", modality="private_test_meta"
            )
        elif private_meta.suffix.lower() == ".jsonl":
            records["private_test_meta"] = read_jsonl_records(private_meta, "private_test_meta")

    return records


def valid_hash_records(records: list[SampleRecord]) -> list[SampleRecord]:
    return [r for r in records if r.hash and r.hash_type != "error"]


def hash_to_records(records: list[SampleRecord]) -> dict[str, list[SampleRecord]]:
    m: dict[str, list[SampleRecord]] = {}
    for r in valid_hash_records(records):
        m.setdefault(r.hash, []).append(r)
    return m


def build_exact_overlap_report(records_by_split: dict[str, list[SampleRecord]]) -> dict[str, Any]:
    train_like = valid_hash_records(records_by_split.get("train", [])) + valid_hash_records(records_by_split.get("val", []))
    test_like = valid_hash_records(records_by_split.get("test", []))
    external = valid_hash_records(records_by_split.get("external", []))
    private_test_meta = valid_hash_records(records_by_split.get("private_test_meta", []))

    train_map = hash_to_records(train_like)
    test_map = hash_to_records(test_like)
    external_map = hash_to_records(external)

    train_test_overlap_hashes = sorted(set(train_map) & set(test_map))
    external_test_overlap_hashes = sorted(set(external_map) & set(test_map))

    train_test_pairs = []
    for h in train_test_overlap_hashes:
        train_examples = train_map[h]
        test_examples = test_map[h]
        train_test_pairs.append({
            "hash": h,
            "train_records": [asdict(x) for x in train_examples[:5]],
            "test_records": [asdict(x) for x in test_examples[:5]],
            "train_match_count": len(train_examples),
            "test_match_count": len(test_examples),
        })

    external_test_pairs = []
    for h in external_test_overlap_hashes:
        external_examples = external_map[h]
        test_examples = test_map[h]
        external_test_pairs.append({
            "hash": h,
            "external_records": [asdict(x) for x in external_examples[:5]],
            "test_records": [asdict(x) for x in test_examples[:5]],
            "external_match_count": len(external_examples),
            "test_match_count": len(test_examples),
        })

    return {
        "train_like_hash_count": len({r.hash for r in train_like}),
        "test_like_hash_count": len({r.hash for r in test_like}),
        "external_hash_count": len({r.hash for r in external}),
        "private_test_meta_hash_count": len({r.hash for r in private_test_meta}),

        "train_sample_record_count": len(records_by_split.get("train", [])),
        "val_sample_record_count": len(records_by_split.get("val", [])),
        "test_sample_record_count": len(records_by_split.get("test", [])),
        "external_sample_record_count": len(records_by_split.get("external", [])),
        "private_test_meta_record_count": len(records_by_split.get("private_test_meta", [])),

        "train_test_overlap_count": len(train_test_overlap_hashes),
        "external_test_overlap_count": len(external_test_overlap_hashes),
        "train_test_overlap_ratio": ratio(len(train_test_overlap_hashes), len({r.hash for r in test_like})),
        "external_test_overlap_ratio": ratio(len(external_test_overlap_hashes), len({r.hash for r in test_like})),

        "train_test_overlap_examples": train_test_pairs[:20],
        "external_test_overlap_examples": external_test_pairs[:20],
    }

def build_overlap_curve(
    records_by_split: dict[str, list[SampleRecord]],
    nn_top_pairs: list[dict[str, Any]] | None = None,
    nn_threshold: float = 0.95,
    num_points: int = 20,
) -> dict[str, Any]:
    """
    x-axis: test dataset size
    y-axis:
      1. exact hash overlap rate against train-like hashes
      2. NN similarity overlap rate against train-like samples
    """
    train_like = (
        valid_hash_records(records_by_split.get("train", []))
        + valid_hash_records(records_by_split.get("val", []))
    )
    test_like = valid_hash_records(records_by_split.get("test", []))

    train_hashes = {r.hash for r in train_like if r.hash}
    test_like = sorted(test_like, key=lambda r: r.sample_id)

    n = len(test_like)
    if n == 0:
        return {
            "dataset_sizes": [],
            "hash_overlap_rates": [],
            "hash_overlap_counts": [],
            "nn_overlap_rates": [],
            "nn_overlap_counts": [],
        }

    if n <= num_points:
        sizes = list(range(1, n + 1))
    else:
        sizes = sorted(set(
            max(1, round(i * n / num_points))
            for i in range(1, num_points + 1)
        ))

    # Build test sample -> NN similarity map from top pairs.
    # This only uses stored top pairs, so if you want exact full-curve NN,
    # make sure nn_top_pairs includes all test samples. For now this works
    # as a visualization approximation.
    nn_sim_by_test = {}
    for pair in nn_top_pairs or []:
        test_key = pair.get("test_path") or pair.get("test_sample")
        sim = pair.get("similarity")
        if test_key is not None and sim is not None:
            nn_sim_by_test[str(test_key)] = float(sim)

    hash_overlap_counts = []
    hash_overlap_rates = []
    nn_overlap_counts = []
    nn_overlap_rates = []

    for size in sizes:
        subset = test_like[:size]

        subset_hashes = {r.hash for r in subset if r.hash}
        hash_overlap_count = len(subset_hashes & train_hashes)
        hash_overlap_rate = hash_overlap_count / len(subset_hashes) if subset_hashes else 0.0

        nn_count = 0
        nn_total = 0
        for r in subset:
            keys = [
                r.source_path,
                r.sample_id,
                str(Path(r.source_path)) if r.source_path else "",
            ]
            matched_sim = None
            for k in keys:
                if k in nn_sim_by_test:
                    matched_sim = nn_sim_by_test[k]
                    break

            if matched_sim is not None:
                nn_total += 1
                if matched_sim >= nn_threshold:
                    nn_count += 1

        nn_rate = nn_count / nn_total if nn_total else 0.0

        hash_overlap_counts.append(hash_overlap_count)
        hash_overlap_rates.append(hash_overlap_rate)
        nn_overlap_counts.append(nn_count)
        nn_overlap_rates.append(nn_rate)

    return {
        "dataset_sizes": sizes,
        "hash_overlap_rates": hash_overlap_rates,
        "hash_overlap_counts": hash_overlap_counts,
        "nn_overlap_rates": nn_overlap_rates,
        "nn_overlap_counts": nn_overlap_counts,
        "nn_threshold": nn_threshold,
    }

def save_overlap_curve_plot(
    run_dir: Path,
    exp_id: str,
    curve: dict[str, Any],
) -> str | None:
    """
    Save hash-vs-NN overlap curve to:
      run_dir/visualize/hash_nn_overlap_curve.png
    """
    if plt is None:
        return None

    sizes = curve.get("dataset_sizes", [])
    hash_rates = curve.get("hash_overlap_rates", [])
    nn_rates = curve.get("nn_overlap_rates", [])

    if not sizes or not hash_rates:
        return None

    visualize_dir = run_dir / "visualize"
    visualize_dir.mkdir(parents=True, exist_ok=True)

    out_path = visualize_dir / "hash_nn_overlap_curve.png"

    plt.figure(figsize=(8.5, 5.2))

    plt.plot(
        sizes,
        hash_rates,
        marker="o",
        linewidth=2,
        label="Exact Hash Overlap Rate",
    )

    if nn_rates:
        plt.plot(
            sizes,
            nn_rates,
            marker="x",
            linewidth=2,
            linestyle="--",
            label=f"NN Similarity Overlap Rate (threshold={curve.get('nn_threshold')})",
        )

    plt.xlabel("Test Dataset Size")
    plt.ylabel("Overlap Rate")
    plt.title(f"Exact Hash vs NN Similarity Overlap\n{exp_id}")
    plt.ylim(0, min(1.0, max(0.05, max(hash_rates + (nn_rates or [0])) * 1.15)))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()

    return str(out_path)
# ============================================================
# Suspicious code scan
# ============================================================

def suspicious_code_scan(code_text: str) -> list[str]:
    hits = []
    for pat in SUSPICIOUS_PATTERNS:
        if re.search(pat, code_text, flags=re.IGNORECASE):
            hits.append(pat)
    return hits


# ============================================================
# NN similarity: image pHash
# ============================================================

def _dct_1d(x: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    x_ext = np.concatenate([x, x[::-1]])
    fft = np.fft.fft(x_ext)
    result = np.real(fft[:n])
    result[0] = result[0] / np.sqrt(2.0)
    return result


def image_phash(path: Path, hash_size: int = 8, highfreq_factor: int = 4) -> np.ndarray:
    if Image is None:
        raise RuntimeError("Missing dependency: Pillow. Install with: pip install pillow")

    img_size = hash_size * highfreq_factor
    img = Image.open(path).convert("L").resize((img_size, img_size))
    pixels = np.asarray(img, dtype=np.float32)

    dct_rows = np.apply_along_axis(_dct_1d, axis=1, arr=pixels)
    dct_2d = np.apply_along_axis(_dct_1d, axis=0, arr=dct_rows)

    low_freq = dct_2d[:hash_size, :hash_size]
    flat = low_freq.flatten()
    med = np.median(flat[1:]) if flat.size > 1 else flat[0]
    bits = (low_freq > med).astype(np.uint8).flatten()
    return bits


def phash_similarity_report(
    train_records: list[SampleRecord],
    test_records: list[SampleRecord],
    threshold: float,
) -> dict[str, Any]:
    train_paths = [Path(r.source_path) for r in train_records if r.source_path and Path(r.source_path).exists()]
    test_paths = [Path(r.source_path) for r in test_records if r.source_path and Path(r.source_path).exists()]

    if not train_paths or not test_paths:
        return empty_nn_report()

    train_hashes = np.stack([image_phash(p) for p in train_paths], axis=0)
    test_hashes = np.stack([image_phash(p) for p in test_paths], axis=0)

    xor = np.not_equal(test_hashes[:, None, :], train_hashes[None, :, :])
    hamming = xor.sum(axis=2).astype(np.float32)
    n_bits = train_hashes.shape[1]

    sims = 1.0 - (hamming / n_bits)
    best_idx = sims.argmax(axis=1)
    best_sim = sims.max(axis=1)

    overlap_count = int(np.sum(best_sim >= threshold))

    top_pairs = []
    sorted_test_idxs = np.argsort(-best_sim)
    for idx in sorted_test_idxs[:20]:
        train_idx = int(best_idx[idx])
        top_pairs.append({
            "test_path": str(test_paths[idx]),
            "nearest_train_path": str(train_paths[train_idx]),
            "similarity": float(best_sim[idx]),
        })

    return {
        "nn_method": "image_phash_hamming",
        "nn_threshold": threshold,
        "nn_train_sample_count": len(train_paths),
        "nn_test_sample_count": len(test_paths),
        "nn_max_similarity": float(np.max(best_sim)),
        "nn_mean_similarity": float(np.mean(best_sim)),
        "nn_overlap_count": overlap_count,
        "nn_overlap_ratio": float(overlap_count / len(best_sim)),
        "nn_top_pairs": top_pairs,
    }


# ============================================================
# NN similarity: audio
# ============================================================

def audio_embedding(path: Path, target_sr: int = 16000, n_mfcc: int = 20) -> np.ndarray:
    if librosa is not None:
        try:
            y, sr = librosa.load(str(path), sr=target_sr, mono=True)
            if y.size == 0:
                return np.zeros(n_mfcc * 2, dtype=np.float32)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
            feat = np.concatenate(
                [mfcc.mean(axis=1), mfcc.std(axis=1)],
                axis=0,
            ).astype(np.float32)
            return feat
        except Exception:
            pass

    # Dependency-light fallback.
    data = path.read_bytes()
    if len(data) == 0:
        return np.zeros(128, dtype=np.float32)

    arr = np.frombuffer(data[: min(len(data), 65536)], dtype=np.uint8).astype(np.float32)
    if arr.size < 128:
        arr = np.pad(arr, (0, 128 - arr.size))
    else:
        usable = 128 * (arr.size // 128)
        arr = arr[:usable].reshape(128, -1).mean(axis=1)

    return arr / 255.0


def cosine_nn_report(
    X_train,
    X_test,
    threshold: float,
    train_labels: list[str] | None = None,
    test_labels: list[str] | None = None,
    method: str = "cosine_nn",
) -> dict[str, Any]:
    if NearestNeighbors is None:
        raise RuntimeError("Missing dependency: scikit-learn. Install with: pip install scikit-learn")

    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        return empty_nn_report(method=method, threshold=threshold)

    nn = NearestNeighbors(n_neighbors=1, metric="cosine")
    nn.fit(X_train)

    distances, indices = nn.kneighbors(X_test)
    sims = 1.0 - distances.reshape(-1)
    nearest = indices.reshape(-1)

    overlap_count = int(np.sum(sims >= threshold))

    top_pairs = []
    sorted_idxs = np.argsort(-sims)
    for idx in sorted_idxs[:20]:
        train_idx = int(nearest[idx])
        top_pairs.append({
            "test_sample": test_labels[idx] if test_labels else str(idx),
            "nearest_train_sample": train_labels[train_idx] if train_labels else str(train_idx),
            "similarity": float(sims[idx]),
        })

    return {
        "nn_method": method,
        "nn_threshold": threshold,
        "nn_train_sample_count": int(X_train.shape[0]),
        "nn_test_sample_count": int(X_test.shape[0]),
        "nn_max_similarity": float(np.max(sims)),
        "nn_mean_similarity": float(np.mean(sims)),
        "nn_overlap_count": overlap_count,
        "nn_overlap_ratio": float(overlap_count / len(sims)),
        "nn_top_pairs": top_pairs,
    }


def audio_similarity_report(
    train_records: list[SampleRecord],
    test_records: list[SampleRecord],
    threshold: float,
) -> dict[str, Any]:
    train_paths = [Path(r.source_path) for r in train_records if Path(r.source_path).exists()]
    test_paths = [Path(r.source_path) for r in test_records if Path(r.source_path).exists()]

    if not train_paths or not test_paths:
        return empty_nn_report(method="audio_embedding_nn", threshold=threshold)

    X_train = np.stack([audio_embedding(p) for p in train_paths], axis=0)
    X_test = np.stack([audio_embedding(p) for p in test_paths], axis=0)

    X_train = l2_normalize(X_train)
    X_test = l2_normalize(X_test)

    return cosine_nn_report(
        X_train,
        X_test,
        threshold=threshold,
        train_labels=[str(p) for p in train_paths],
        test_labels=[str(p) for p in test_paths],
        method="audio_embedding_cosine_nn",
    )


# ============================================================
# NN similarity: text / tabular
# ============================================================

def serialize_record_for_text_nn(record: SampleRecord) -> str:
    if record.modality in {"vision", "audio"}:
        return record.sample_id
    return json.dumps(
        {
            "sample_id": record.sample_id,
            "source_path": record.source_path,
            "hash": record.hash,
            "extra": record.extra,
        },
        sort_keys=True,
        ensure_ascii=False,
    )


def build_text_nn_strings_from_table(path: Path, max_rows: int) -> list[str]:
    rows = read_manifest_rows(path)
    strings = []
    for row in rows[:max_rows]:
        clean = {k: v for k, v in row.items() if k != "_row_idx"}
        strings.append(json.dumps(clean, sort_keys=True, ensure_ascii=False))
    return strings


def build_text_nn_strings_from_path(path: Path, max_rows: int) -> list[str]:
    if not path.exists() or not path.is_file():
        return []

    suffix = path.suffix.lower()

    if suffix in {".csv", ".tsv"}:
        return build_text_nn_strings_from_table(path, max_rows)

    if suffix == ".jsonl":
        strings = []
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line:
                    strings.append(line)
                if len(strings) >= max_rows:
                    break
        return strings

    if suffix in {".txt", ".md", ".json"}:
        strings = []
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line:
                    strings.append(line)
                if len(strings) >= max_rows:
                    break
        return strings

    return []


def text_or_tabular_similarity_report(
    split_paths: dict[str, Path],
    threshold: float,
    max_train: int,
    max_test: int,
    method: str,
) -> dict[str, Any]:
    if TfidfVectorizer is None:
        raise RuntimeError("Missing dependency: scikit-learn. Install with: pip install scikit-learn")

    train_strings: list[str] = []
    for split in ["train", "val"]:
        p = split_paths.get(split)
        if p is not None:
            train_strings.extend(build_text_nn_strings_from_path(p, max_train))

    test_strings: list[str] = []
    p = split_paths.get("test")
    if p is not None:
        test_strings.extend(build_text_nn_strings_from_path(p, max_test))

    train_strings = deterministic_sample(train_strings, max_train)
    test_strings = deterministic_sample(test_strings, max_test)

    if not train_strings or not test_strings:
        return empty_nn_report(method=method, threshold=threshold)

    vectorizer = TfidfVectorizer(
        lowercase=True,
        max_features=20000,
        ngram_range=(1, 2),
        min_df=1,
    )

    X_train = vectorizer.fit_transform(train_strings)
    X_test = vectorizer.transform(test_strings)

    return cosine_nn_report(
        X_train,
        X_test,
        threshold=threshold,
        train_labels=[f"train_like_{i}" for i in range(len(train_strings))],
        test_labels=[f"test_{i}" for i in range(len(test_strings))],
        method=method,
    )


def empty_nn_report(method: str = "none", threshold: float | None = None) -> dict[str, Any]:
    return {
        "nn_method": method,
        "nn_threshold": threshold,
        "nn_train_sample_count": 0,
        "nn_test_sample_count": 0,
        "nn_max_similarity": None,
        "nn_mean_similarity": None,
        "nn_overlap_count": 0,
        "nn_overlap_ratio": 0.0,
        "nn_top_pairs": [],
    }


def build_nn_report(
    effective_mode: str,
    records_by_split: dict[str, list[SampleRecord]],
    split_paths: dict[str, Path],
    threshold: float,
    max_train: int,
    max_test: int,
) -> dict[str, Any]:
    train_like = valid_hash_records(records_by_split.get("train", [])) + valid_hash_records(records_by_split.get("val", []))
    test_like = valid_hash_records(records_by_split.get("test", []))

    train_like = deterministic_sample(train_like, max_train)
    test_like = deterministic_sample(test_like, max_test)

    if effective_mode in {"vision", "vision_via_manifest"}:
        return phash_similarity_report(train_like, test_like, threshold)

    if effective_mode in {"audio", "audio_via_manifest"}:
        return audio_similarity_report(train_like, test_like, threshold)

    if effective_mode == "tabular":
        return text_or_tabular_similarity_report(
            split_paths,
            threshold=threshold,
            max_train=max_train,
            max_test=max_test,
            method="tabular_row_tfidf_cosine_nn",
        )

    if effective_mode in {"text", "text_via_manifest"}:
        return text_or_tabular_similarity_report(
            split_paths,
            threshold=threshold,
            max_train=max_train,
            max_test=max_test,
            method="text_tfidf_cosine_nn",
        )

    return empty_nn_report(threshold=threshold)


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--best-node-id", required=True, type=str)
    parser.add_argument("--nn-threshold", type=float, default=0.95)
    parser.add_argument("--max-nn-train", type=int, default=5000)
    parser.add_argument("--max-nn-test", type=int, default=2000)
    parser.add_argument(
        "--write-sample-hashes",
        action="store_true",
        help="Write detailed per-sample hash records to JSONL files.",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    best_node_id = args.best_node_id.strip()

    cfg_path = find_run_config(run_dir)
    cfg = parse_config(cfg_path)

    exp_id, dataset_root = infer_dataset_root(cfg)

    category = MLE_LITE_CATEGORIES.get(exp_id, "unknown")
    category_modality = category_to_modality(category)

    best_code_path = find_best_code_path(run_dir, best_node_id)
    best_code = safe_read_text(best_code_path)
    code_mode = detect_code_mode(best_code)
    effective_mode = decide_effective_mode(category_modality, code_mode)

    split_paths = infer_split_paths(dataset_root)
    image_asset_roots = find_asset_roots(dataset_root, IMAGE_EXTS)
    audio_asset_roots = find_asset_roots(dataset_root, AUDIO_EXTS)
    text_asset_roots = find_asset_roots(dataset_root, TEXT_EXTS)

    records_by_split = build_sample_records(
        effective_mode=effective_mode,
        split_paths=split_paths,
        image_asset_roots=image_asset_roots,
        audio_asset_roots=audio_asset_roots,
    )

    exact_report = build_exact_overlap_report(records_by_split)

    nn_report = build_nn_report(
    effective_mode=effective_mode,
    records_by_split=records_by_split,
    split_paths=split_paths,
    threshold=args.nn_threshold,
    max_train=args.max_nn_train,
    max_test=args.max_nn_test,
    )

    overlap_curve = build_overlap_curve(
    records_by_split=records_by_split,
    nn_top_pairs=nn_report.get("nn_top_pairs", []),
    nn_threshold=args.nn_threshold,
    )

    overlap_curve_plot_path = save_overlap_curve_plot(
    run_dir=run_dir,
    exp_id=exp_id,
    curve=overlap_curve,
    )

    suspicious_hits = suspicious_code_scan(best_code)

    # Every test sample has its own hash record here.
    test_hash_records = [asdict(r) for r in records_by_split.get("test", [])]

    report = {
        "run_dir": str(run_dir),
        "dataset_root": str(dataset_root),
        "exp_id": exp_id,
        "category": category,
        "category_modality": category_modality,
        "best_node_id": best_node_id,
        "best_code_path": str(best_code_path),

        "code_mode": code_mode,
        "effective_mode": effective_mode,

        "split_paths": {k: str(v) for k, v in split_paths.items()},
        "image_asset_roots_sample": [str(p) for p in image_asset_roots[:10]],
        "audio_asset_roots_sample": [str(p) for p in audio_asset_roots[:10]],
        "text_asset_roots_sample": [str(p) for p in text_asset_roots[:10]],

        **exact_report,
        **nn_report,

        "test_sample_hash_records_count": len(test_hash_records),
        "test_sample_hash_records_preview": test_hash_records[:20],

        "suspicious_code_patterns": suspicious_hits,

        "hash_overlap_curve": overlap_curve,
        "hash_overlap_curve_plot_path": overlap_curve_plot_path,
    }

    out_path = run_dir / "mle_lite_best_node_leakage_report.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.write_sample_hashes:
        hash_dir = run_dir / "mle_lite_sample_hashes"
        hash_dir.mkdir(parents=True, exist_ok=True)

        for split, records in records_by_split.items():
            out = hash_dir / f"{split}_hashes.jsonl"
            with open(out, "w", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    print("===== MLE-Lite Best Node Leakage Report =====")
    print(f"run_dir: {run_dir}")
    print(f"dataset_root: {dataset_root}")
    print(f"exp_id: {exp_id}")
    print(f"category: {category}")
    print(f"category_modality: {category_modality}")
    print(f"best_node_id: {best_node_id}")
    print(f"best_code_path: {best_code_path}")
    print(f"code_mode: {code_mode}")
    print(f"effective_mode: {effective_mode}")
    print(f"split_paths: {report['split_paths']}")

    print("")
    print("----- Exact Hash Leakage -----")
    print(f"train_sample_record_count: {report['train_sample_record_count']}")
    print(f"val_sample_record_count: {report['val_sample_record_count']}")
    print(f"test_sample_record_count: {report['test_sample_record_count']}")
    print(f"external_sample_record_count: {report['external_sample_record_count']}")
    print(f"private_test_meta_record_count: {report['private_test_meta_record_count']}")
    print(f"train_like_hash_count: {report['train_like_hash_count']}")
    print(f"test_like_hash_count: {report['test_like_hash_count']}")
    print(f"external_hash_count: {report['external_hash_count']}")
    print(f"private_test_meta_hash_count: {report['private_test_meta_hash_count']}")
    print(f"train_test_overlap_count: {report['train_test_overlap_count']}")
    print(f"external_test_overlap_count: {report['external_test_overlap_count']}")
    print(f"train_test_overlap_ratio: {report['train_test_overlap_ratio']:.6f}")
    print(f"external_test_overlap_ratio: {report['external_test_overlap_ratio']:.6f}")

    print("")
    print("----- NN Similarity -----")
    print(f"nn_method: {report['nn_method']}")
    print(f"nn_threshold: {report['nn_threshold']}")
    print(f"nn_train_sample_count: {report['nn_train_sample_count']}")
    print(f"nn_test_sample_count: {report['nn_test_sample_count']}")
    print(f"nn_max_similarity: {report['nn_max_similarity']}")
    print(f"nn_mean_similarity: {report['nn_mean_similarity']}")
    print(f"nn_overlap_count: {report['nn_overlap_count']}")
    print(f"nn_overlap_ratio: {report['nn_overlap_ratio']}")

    print("")
    print("----- Code Risk Scan -----")
    print(f"suspicious_code_patterns: {suspicious_hits}")

    print("")
    print(f"test_sample_hash_records_count: {len(test_hash_records)}")
    print(f"hash_nn_overlap_curve_plot_path: {overlap_curve_plot_path}")
    print(f"saved_report: {out_path}")

    if args.write_sample_hashes:
        print(f"saved_sample_hashes_dir: {run_dir / 'mle_lite_sample_hashes'}")


if __name__ == "__main__":
    main()