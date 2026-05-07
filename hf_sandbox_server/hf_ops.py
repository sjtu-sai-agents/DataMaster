"""Core HuggingFace operations – extracted from search_huggingface_math_posttrain.py.

These functions run inside the sandbox and make the actual HF API calls.
They are called by server.py after rate-limiting and concurrency control.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from math import ceil
from pathlib import Path
from typing import Any, Optional

import requests
import yaml
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from . import config as cfg

LOGGER = logging.getLogger(__name__)

# ── Session management ───────────────────────────────────────────────

_SESSION: requests.Session | None = None
SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _get_session() -> requests.Session:
    global _SESSION
    if _SESSION is not None:
        return _SESSION
    session = requests.Session()
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=1.0,
        status_forcelist=(408, 429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "HEAD"}),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=8, pool_maxsize=8)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    _SESSION = session
    return session


def _hf_endpoint() -> str:
    return cfg.HF_ENDPOINT


def _hf_headers() -> dict[str, str]:
    headers = {
        "Accept": "application/json",
        "User-Agent": "hf-sandbox-server/1.0",
    }
    if cfg.HF_TOKEN:
        headers["Authorization"] = f"Bearer {cfg.HF_TOKEN}"
    return headers


def _request_json(path: str, *, params: dict[str, Any] | None = None, timeout: int = 25) -> Any:
    url = f"{_hf_endpoint()}{path}"
    resp = _get_session().get(url, headers=_hf_headers(), params=params or {}, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _request_text(url: str, *, timeout: int = 25) -> str:
    resp = _get_session().get(url, headers=_hf_headers(), timeout=timeout)
    resp.raise_for_status()
    return resp.text


def _safe_source_filename(source_id: str) -> str:
    return SAFE_FILENAME_RE.sub("_", source_id).strip("._") or "dataset"


# ── Search ───────────────────────────────────────────────────────────

def search_datasets(query: str, limit: int = 100, author: str | None = None) -> str:
    """Return formatted search results text (same format as MCP tool)."""
    from huggingface_hub import list_datasets as _list_datasets

    try:
        payload = _request_json(
            "/api/datasets",
            params={
                "search": f"{query} author:{author}" if author else query,
                "limit": limit,
            },
            timeout=25,
        )
        results = payload if isinstance(payload, list) else []
        if not results:
            return f"No accessible datasets found for query: '{query}'"

        lines = [f"Found {len(results)} datasets for '{query}':", ""]
        for i, item in enumerate(results, 1):
            if not isinstance(item, dict):
                continue
            dataset_id = str(item.get("id") or item.get("name") or "unknown")
            author_name = item.get("author")
            if not author_name and "/" in dataset_id:
                author_name = dataset_id.split("/", 1)[0]
            downloads = item.get("downloads") or 0
            likes = item.get("likes") or 0
            lines.append(
                f"{i}. {dataset_id} | Author: {author_name or 'unknown'} | "
                f"Downloads: {downloads:,} | Likes: {likes} | "
                f"URL: https://huggingface.co/datasets/{dataset_id}"
            )
        return "\n".join(lines)
    except Exception as api_error:
        try:
            search_params = {
                "search": f"{query} author:{author}" if author else query,
                "limit": limit,
            }
            results = list(_list_datasets(**search_params))
            if not results:
                return f"No accessible datasets found for query: '{query}'"
            lines = [f"Found {len(results)} datasets for '{query}':", ""]
            for i, ds in enumerate(results, 1):
                lines.append(
                    f"{i}. {ds.id} | Author: {ds.author or 'unknown'} | "
                    f"Downloads: {ds.downloads or 0:,} | Likes: {ds.likes or 0} | "
                    f"URL: https://huggingface.co/datasets/{ds.id}"
                )
            return "\n".join(lines)
        except Exception as hub_error:
            return (
                f"Error searching datasets: REST API failed with {api_error}; "
                f"huggingface_hub fallback failed with {hub_error}"
            )


# ── Inspect ──────────────────────────────────────────────────────────

def inspect_dataset(dataset_id: str, config: str | None = None) -> str:
    from huggingface_hub import dataset_info as _dataset_info
    from datasets import load_dataset_builder, get_dataset_config_names

    hub_info = None
    try:
        from huggingface_hub import login
        token = cfg.HF_TOKEN
        if token:
            try:
                login(token=token)
            except Exception:
                pass
        hub_info = _dataset_info(dataset_id)
    except Exception:
        pass

    builder = None
    try:
        builder = load_dataset_builder(dataset_id, name=config)
    except Exception:
        pass

    configs = []
    try:
        configs = get_dataset_config_names(dataset_id)
    except Exception:
        pass

    hub_metadata = {}
    if hub_info:
        hub_metadata = {
            "id": hub_info.id,
            "author": hub_info.author,
            "created_at": hub_info.created_at.isoformat() if hub_info.created_at else None,
            "last_modified": hub_info.last_modified.isoformat() if hub_info.last_modified else None,
            "private": hub_info.private,
            "disabled": hub_info.disabled,
            "gated": hub_info.gated,
            "downloads": hub_info.downloads,
            "downloads_all_time": hub_info.downloads_all_time,
            "likes": hub_info.likes,
            "tags": hub_info.tags,
            "paperswithcode_id": hub_info.paperswithcode_id,
            "trending_score": hub_info.trending_score,
            "files": [s.rfilename for s in hub_info.siblings],
        }

    builder_info = {}
    if builder:
        info = builder.info
        builder_info = {
            "dataset_name": builder.dataset_name,
            "config_name": builder.name,
            "description": info.description,
            "citation": info.citation,
            "homepage": info.homepage,
            "license": info.license,
            "features": {k: str(v) for k, v in info.features.items()} if info.features else {},
            "splits": {k: str(v) for k, v in info.splits.items()} if info.splits else {},
            "configs": configs,
        }

    merged = {**hub_metadata, **builder_info}
    return yaml.dump(merged, allow_unicode=True, sort_keys=False, default_flow_style=False)


# ── Configs / Splits ─────────────────────────────────────────────────

def get_dataset_configs(dataset_id: str) -> str:
    from datasets import get_dataset_config_names as _get_configs
    try:
        configs = _get_configs(dataset_id)
        if not configs:
            return "This dataset uses the default configuration (no multiple configs)."
        return json.dumps(configs)
    except Exception as e:
        return f"Error listing configs for '{dataset_id}': {e}"


def get_dataset_splits(dataset_id: str, config: str | None = None) -> str:
    from datasets import load_dataset as _load_dataset
    try:
        ds = _load_dataset(dataset_id, name=config)
        return str(ds)
    except Exception as e:
        return f"Error listing splits for '{dataset_id}': {e}"


# ── README ───────────────────────────────────────────────────────────

def get_dataset_readme(dataset_id: str) -> str:
    from huggingface_hub import dataset_info as _dataset_info, hf_hub_download as _download

    errors: list[str] = []

    try:
        content = _request_text(
            f"{_hf_endpoint()}/datasets/{dataset_id}/resolve/main/README.md", timeout=25
        )
        if content.strip():
            return content
    except Exception as e:
        errors.append(f"direct_readme_get={e}")

    try:
        payload = _request_json(f"/api/datasets/{dataset_id}", timeout=25)
        if isinstance(payload, dict):
            card = payload.get("cardData") or {}
            if isinstance(card, dict):
                desc = card.get("description", "") or (card.get("dataset_info", {}) or {}).get("description", "")
                if desc:
                    return desc
    except Exception as e:
        errors.append(f"api_dataset_info={e}")

    try:
        info = _dataset_info(dataset_id)
        desc = info.card_data.get("description", "") or info.card_data.get("dataset_info", {}).get("description", "")
        if desc:
            return desc
    except Exception as e:
        errors.append(f"huggingface_hub_dataset_info={e}")

    try:
        readme_path = _download(repo_id=dataset_id, filename="README.md", repo_type="dataset")
        with open(readme_path, "r", encoding="utf-8") as f:
            content = f.read()
        if content.strip():
            return content
    except Exception as e:
        errors.append(f"hf_hub_download={e}")

    return f"Error retrieving README for '{dataset_id}': {' | '.join(errors) if errors else 'not found'}"


# ── Sample ───────────────────────────────────────────────────────────

def get_dataset_sample(
    dataset_id: str,
    config: str | None = None,
    split: str | None = None,
    num_samples: int = 5,
) -> str:
    from datasets import load_dataset as _load_dataset, get_dataset_split_names as _get_splits

    try:
        split_names = _get_splits(dataset_id, config_name=config)
        if split and split in split_names:
            pass
        else:
            split = split_names[0] if split_names else "train"

        ds = _load_dataset(dataset_id, name=config, split=split)
        samples = []
        for i, example in enumerate(ds):
            if i >= num_samples:
                break
            samples.append({
                k: v if isinstance(v, (str, int, float, bool, list, dict)) or v is None else str(v)
                for k, v in example.items()
            })

        if not samples:
            return f"No samples retrieved from '{dataset_id}' (split: '{split}')"

        columns = list(samples[0].keys())
        msgs = [
            f"Dataset Sample from '{dataset_id}'",
            f"Config: {config or 'default'} | Split: {split}",
            f"Columns ({len(columns)}): {', '.join(columns)}",
            "\nColumn Types:",
        ]
        msgs.extend(f"  {k}: {type(v).__name__}" for k, v in samples[0].items())
        for i, s in enumerate(samples, 1):
            msgs.append(f"\n[Sample {i}]")
            for k, v in s.items():
                msgs.append(f"  {k}: {v}")
        return "\n".join(msgs)
    except Exception as e:
        return f"Error calling tools: {e}"


# ── Download ─────────────────────────────────────────────────────────

def download_dataset(dataset_id: str, output_dir: str = "./downloaded_datasets") -> str:
    from huggingface_hub import HfFileSystem, hf_hub_download as _download

    try:
        token = cfg.HF_TOKEN or None
        fs = HfFileSystem(token=token)
        repo_path = f"datasets/{dataset_id}"
        try:
            all_files = list(fs.ls(repo_path, recursive=True))
        except Exception:
            return "Download Error: failed to list files"

        results = []
        for fp in all_files:
            if fp["type"] != "file":
                continue
            fname = fp["name"]
            try:
                local = _download(
                    repo_id=dataset_id,
                    filename=fname.replace(f"datasets/{dataset_id}/", ""),
                    repo_type="dataset",
                    token=token,
                    local_dir=output_dir,
                    local_dir_use_symlinks=False,
                )
                size_mb = round(os.path.getsize(local) / (1024 * 1024), 2)
                results.append(f"Raw file: {os.path.basename(local)} ({size_mb} MB) -> {local}")
            except Exception as e:
                results.append(f"Failed to download {fname}: {e}")

        if results:
            return "\n".join([
                f"Raw File Download: '{dataset_id}'",
                f"Output: {os.path.abspath(output_dir)}",
                "--- Downloaded Files ---",
                *results,
                f"\nURL: https://huggingface.co/datasets/{dataset_id}",
            ])
        return "Download Error"
    except Exception:
        return "Download Error"


# ── Materialize ──────────────────────────────────────────────────────

def materialize_dataset(
    dataset: str,
    config: str | None = None,
    split: str | None = None,
    max_rows: int = 2048,
    output_dir: str = "",
) -> dict[str, Any]:
    """Materialize a HuggingFace dataset to JSONL on shared NFS.

    Returns {"local_path": "<path>", "rows": <n>} on success,
    or {"error": "<msg>"} on failure.
    """
    from datasets import load_dataset as _load_dataset, get_dataset_config_names, get_dataset_split_names

    cache_root = Path(cfg.SHARED_CACHE_DIR) / "materialized_datasets"
    cache_root.mkdir(parents=True, exist_ok=True)

    config_name = (config or "default").strip()
    max_rows_tag = "all" if max_rows is None else str(max_rows)
    cache_key = f"{_safe_source_filename(dataset)}_{config_name}_{max_rows_tag}"
    output_path = cache_root / f"{cache_key}.jsonl"

    if output_path.exists() and output_path.stat().st_size > 0:
        row_count = sum(1 for line in output_path.open() if line.strip())
        if row_count > 0:
            return {"local_path": str(output_path), "rows": row_count}

    config_names: list[str] = []
    try:
        config_names = list(get_dataset_config_names(dataset))
    except Exception:
        pass

    if config and config != "default":
        configs_to_try = [config]
    elif config_names:
        configs_to_try = config_names
    else:
        configs_to_try = [""]

    split_pref = (split or "").strip()

    rows: list[dict[str, Any]] = []
    per_config_budget = max_rows if len(configs_to_try) <= 1 else max(1, ceil(max_rows / len(configs_to_try)))

    for cfg_name in configs_to_try:
        remaining = max_rows - len(rows) if max_rows else None
        if remaining is not None and remaining <= 0:
            break
        budget = min(remaining, per_config_budget) if remaining is not None else per_config_budget

        try:
            split_names = list(get_dataset_split_names(dataset, config_name=cfg_name or None))
        except Exception:
            split_names = []

        chosen_split = _pick_split(split_names, split_pref)
        if not chosen_split:
            chosen_split = split_pref or "train"

        load_args = [dataset]
        if cfg_name:
            load_args.append(cfg_name)
        try:
            slice_spec = f"{chosen_split}[:{budget}]" if budget else chosen_split
            ds = _load_dataset(*load_args, split=slice_spec)
        except Exception:
            try:
                ds = _load_dataset(*load_args, split=chosen_split)
            except Exception as e:
                LOGGER.warning("Failed to load %s config=%s split=%s: %s", dataset, cfg_name, chosen_split, e)
                continue

        for idx, row in enumerate(ds):
            if budget and idx >= budget:
                break
            if isinstance(row, dict):
                if cfg_name:
                    row.setdefault("dataset_config", cfg_name)
                rows.append(row)

    if not rows:
        return {"error": f"No rows materialized for {dataset}"}

    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    return {"local_path": str(output_path), "rows": len(rows)}


def _pick_split(split_names: list[str], preferred: str) -> str:
    if preferred and preferred in split_names:
        return preferred
    train_like = ["train", "training", "train_sft"]
    for name in train_like:
        if name in split_names:
            return name
    for name in split_names:
        lower = name.lower()
        if "train" in lower:
            return name
    skip = {"test", "eval", "evaluation", "dev"}
    for name in split_names:
        if name.lower() not in skip:
            return name
    return split_names[0] if split_names else ""
