from __future__ import annotations

import csv
import hashlib
import importlib.util
import inspect
import json
import logging
import os
import re
import time
from collections import Counter, defaultdict
from math import ceil
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .io import json_dumps_safe, make_json_serializable, read_jsonl, write_json, write_jsonl
from .llama_factory import ALLOWED_HPARAM_OVERRIDES
from .types import MathTrainingExample, TrainPackManifest


DEFAULT_INSTRUCTION = (
    "Solve the following math problem carefully. "
    "Use clear reasoning when helpful and end with the final answer explicitly."
)
DEFAULT_CODE_INSTRUCTION = (
    "Write a correct solution for the coding task. "
    "Return the final code directly."
)
DEFAULT_TOOL_USE_INSTRUCTION = (
    "You are a tool-using assistant. Follow the provided system prompt, tool schema, and prior conversation. "
    "Produce the exact next assistant message, preserving any required tool-call markup."
)
CODE_GENERATION_TASK_TYPES = {"code_generation", "coding", "programming", "code"}
TOOL_USE_TASK_TYPES = {"function_calling", "api_calling", "tool_using", "tool_calling"}

BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
WHITESPACE_RE = re.compile(r"\s+")
FINAL_ANSWER_RE = re.compile(
    r"(?:final answer|answer)\s*[:：-]?\s*(.+)$",
    flags=re.IGNORECASE | re.DOTALL,
)
TRAILING_MATH_TOKEN_RE = re.compile(
    r"(-?\d+(?:/\d+)?(?:\.\d+)?|[A-Za-z]\w*)\s*$"
)
SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
AVAILABLE_CONFIGS_RE = re.compile(r"Available configs(?: in the cache)?\s*:\s*\[([^\]]+)\]")
DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"
DEFAULT_DATASETS_SERVER_BASE = "https://datasets-server.huggingface.co"
DATASETS_SERVER_RETRIES = 4
DATASETS_SERVER_BACKOFF_SECONDS = 1.0
MIN_REQUEST_INTERVAL_SECONDS = 0.2  # Minimum time between requests to avoid overwhelming server
MATERIALIZATION_MIN_ROWS = 512
MATERIALIZATION_OVERSAMPLE_FACTOR = 4
MATERIALIZATION_HARD_CAP = 20_000
DEFAULT_PROBE_MAX_ROWS = 24
DEFAULT_PROBE_PREVIEW_ROWS = 3
DEFAULT_ADAPTER_REPAIR_MIN_FILTERED_RATE = 0.10
DIRECT_TRAIN_CONFIG_DEFAULTS = {
    "num_train_epochs": 1,
    "learning_rate": 1e-4,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 8,
    "cutoff_len": 4096,
    "max_samples": 1000,
}
DIRECT_TRAIN_CONFIG_LIMITS = {
    "num_train_epochs": (0.25, 8.0),
    "learning_rate": (1e-6, 5e-4),
    "per_device_train_batch_size": (1, 256),
    "gradient_accumulation_steps": (1, 128),
    "cutoff_len": (256, 16384),
    "max_samples": (1, 5000),
}
TRAIN_CONFIG_ALLOWED_FIELDS = {
    key for key in ALLOWED_HPARAM_OVERRIDES
    if key in DIRECT_TRAIN_CONFIG_DEFAULTS
}
LOGGER = logging.getLogger(__name__)


# ── Sandbox 代理 (可选) ──────────────────────────────────────────────
_HF_SANDBOX_URL = os.getenv("HF_SANDBOX_URL", "").rstrip("/")


def _sandbox_materialize(
    entry: dict[str, Any],
    max_rows: int | None,
) -> str:
    """Materialize via the HF sandbox service. Returns local_path or empty string."""
    if not _HF_SANDBOX_URL:
        return ""
    dataset_id = _extract_hf_dataset_id(entry)
    if not dataset_id:
        return ""
    config = str(
        entry.get("config") or entry.get("config_name")
        or entry.get("subset") or entry.get("dataset_config") or ""
    ).strip() or None
    split = str(entry.get("split") or "").strip() or None
    try:
        import httpx
        resp = httpx.post(
            f"{_HF_SANDBOX_URL}/materialize",
            json={
                "dataset": dataset_id,
                "config": config,
                "split": split,
                "max_rows": max_rows or 2048,
            },
            timeout=600,
        )
        if resp.status_code == 200:
            data = resp.json()
            path = data.get("local_path", "")
            if path and Path(path).exists():
                LOGGER.info("Materialized via sandbox: %s -> %s", dataset_id, path)
                return path
    except Exception as e:
        LOGGER.warning("sandbox materialize failed for %s, falling back to direct: %s", dataset_id, e)
    return ""


# Global session for connection pooling and reuse
_DATASETS_SERVER_SESSION: requests.Session | None = None
_LAST_REQUEST_TIME: float = 0.0


def normalize_data_access_config(config: dict[str, Any] | None) -> dict[str, Any]:
    raw = dict(config or {})
    datasets_server = dict(raw.get("datasets_server") or {})
    return {
        "hf_endpoint": str(raw.get("hf_endpoint") or os.getenv("HF_ENDPOINT", DEFAULT_HF_ENDPOINT)).rstrip("/"),
        "datasets_server": {
            "enabled": bool(datasets_server.get("enabled", False)),
            "base_url": str(datasets_server.get("base_url") or DEFAULT_DATASETS_SERVER_BASE).rstrip("/"),
        },
    }


def _create_datasets_server_session() -> requests.Session:
    """
    Create a requests session with connection pooling and retry logic.

    Benefits:
    - Reuses TCP/SSL connections (reduces handshake overhead)
    - Automatic retry with exponential backoff
    - Better handling of transient errors
    """
    session = requests.Session()

    # Configure retry strategy for HTTP-level errors
    retry_strategy = Retry(
        total=DATASETS_SERVER_RETRIES,
        backoff_factor=DATASETS_SERVER_BACKOFF_SECONDS,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )

    # Mount adapter with retry strategy and connection pooling
    adapter = HTTPAdapter(
        max_retries=retry_strategy,
        pool_connections=10,
        pool_maxsize=20,
    )

    session.mount("https://", adapter)
    session.mount("http://", adapter)

    # Set default headers
    session.headers.update({
        'Connection': 'keep-alive',
        'User-Agent': 'math-posttrain-datatree/1.0',
    })

    return session


def _get_datasets_server_session() -> requests.Session:
    """Get or create the global datasets-server session."""
    global _DATASETS_SERVER_SESSION
    if _DATASETS_SERVER_SESSION is None:
        _DATASETS_SERVER_SESSION = _create_datasets_server_session()
    return _DATASETS_SERVER_SESSION


def _rate_limit_request() -> None:
    """
    Enforce minimum time between requests to avoid overwhelming the server.
    Helps prevent SSL EOF errors caused by too many concurrent connections.
    """
    global _LAST_REQUEST_TIME
    current_time = time.time()
    time_since_last = current_time - _LAST_REQUEST_TIME

    if time_since_last < MIN_REQUEST_INTERVAL_SECONDS:
        sleep_time = MIN_REQUEST_INTERVAL_SECONDS - time_since_last
        time.sleep(sleep_time)

    _LAST_REQUEST_TIME = time.time()


def normalize_text(text: str) -> str:
    return WHITESPACE_RE.sub(" ", (text or "").strip())


def normalize_final_answer(text: str) -> str:
    raw = (text or "").strip()
    if not raw:
        return ""
    boxed = BOXED_RE.findall(raw)
    if boxed:
        raw = boxed[-1]
    else:
        final_match = FINAL_ANSWER_RE.search(raw)
        if final_match:
            raw = final_match.group(1).strip()
    raw = raw.replace("$", "").replace("**", "").strip()
    raw = re.sub(r"^final answer\s*[:：-]\s*", "", raw, flags=re.IGNORECASE)
    raw = raw.rstrip(".。")
    raw = normalize_text(raw)
    if raw.lower().startswith("the answer is "):
        raw = raw[14:].strip()
    trailing_match = TRAILING_MATH_TOKEN_RE.search(raw)
    if trailing_match and len(raw.split()) > 1:
        raw = trailing_match.group(1)
    return raw


def _pick_first(record: dict[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        value = record.get(key)
        if value is None:
            continue
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _env_flag(name: str) -> bool:
    return str(os.getenv(name) or "").strip().lower() in {"1", "true", "yes", "on"}


def _normalize_label(value: Any) -> str:
    return normalize_text(str(value or "")).lower()


def _derive_answer_style(solution_text: str) -> str:
    return "long_reasoning" if len(solution_text) > 220 or "\n" in solution_text else "short_answer"


def _normalize_code_answer(text: str) -> str:
    return str(text or "").strip()


def _safe_source_filename(source_id: str) -> str:
    return SAFE_FILENAME_RE.sub("_", source_id).strip("._") or "dataset"


def _looks_like_probe_path(path_value: str | Path | None) -> bool:
    path = Path(path_value) if path_value else None
    if path is None:
        return False
    return path.name.endswith("_sample_rows.jsonl") or "_probe" in path.parts


def _get_entry_full_local_path(entry: dict[str, Any]) -> str:
    full_local_path = str(entry.get("full_local_path") or "").strip()
    if full_local_path and not _looks_like_probe_path(full_local_path):
        return full_local_path
    legacy_local_path = str(entry.get("local_path") or "").strip()
    if legacy_local_path and not _looks_like_probe_path(legacy_local_path):
        return legacy_local_path
    return ""


def _set_entry_full_local_path(entry: dict[str, Any], path_value: str) -> None:
    normalized = str(path_value or "").strip()
    entry["full_local_path"] = normalized
    # Keep legacy local_path in sync until all downstream consumers are migrated.
    entry["local_path"] = normalized


def _get_entry_probe_sample_rows_path(entry: dict[str, Any]) -> str:
    return str(entry.get("probe_sample_rows_path") or "").strip()


def _set_entry_probe_sample_rows_path(entry: dict[str, Any], path_value: str) -> None:
    entry["probe_sample_rows_path"] = str(path_value or "").strip()


def _extract_hf_dataset_id(entry: dict[str, Any]) -> str:
    for key in ("source_id", "name"):
        value = str(entry.get(key) or "").strip()
        if value and "/" in value and not value.startswith(("http://", "https://")):
            return value
    url = str(entry.get("url") or "").strip()
    if not url:
        return ""
    parsed = urlparse(url)
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) >= 3 and parts[0] == "datasets":
        return f"{parts[1]}/{parts[2]}"
    return ""


def _datasets_server_get(
    endpoint: str,
    *,
    params: dict[str, Any],
    timeout: int,
    dataset_id: str,
    base_url: str = DEFAULT_DATASETS_SERVER_BASE,
) -> requests.Response:
    """
    Make a GET request to datasets-server with improved error handling.

    Improvements over the previous version:
    - Uses session with connection pooling (reduces SSL handshake overhead)
    - Rate limiting to avoid overwhelming the server
    - Separate connect and read timeouts
    - Special handling for SSL errors with longer backoff

    Args:
        endpoint: API endpoint (e.g., "rows", "splits")
        params: Query parameters
        timeout: Read timeout in seconds
        dataset_id: Dataset identifier for logging

    Returns:
        requests.Response object

    Raises:
        Exception: If all retries are exhausted
    """
    session = _get_datasets_server_session()

    # Use separate connect and read timeouts
    timeout_tuple = (10, timeout)  # (connect_timeout, read_timeout)

    last_error: Exception | None = None
    for attempt in range(1, DATASETS_SERVER_RETRIES + 1):
        try:
            # Rate limit to avoid overwhelming the server
            _rate_limit_request()

            resp = session.get(
                f"{base_url.rstrip('/')}/{endpoint}",
                params=params,
                timeout=timeout_tuple,
            )
            resp.raise_for_status()
            return resp
        except requests.exceptions.SSLError as exc:
            # SSL errors get special treatment with longer backoff
            last_error = exc
            if attempt >= DATASETS_SERVER_RETRIES:
                break
            backoff = DATASETS_SERVER_BACKOFF_SECONDS * attempt * 2
            LOGGER.info(
                "datasets-server %s SSL error for %s (attempt %d/%d): %s; retrying after %.1fs",
                endpoint,
                dataset_id,
                attempt,
                DATASETS_SERVER_RETRIES,
                exc,
                backoff,
            )
            time.sleep(backoff)
        except Exception as exc:
            last_error = exc
            if attempt >= DATASETS_SERVER_RETRIES:
                break
            backoff = DATASETS_SERVER_BACKOFF_SECONDS * attempt
            LOGGER.info(
                "datasets-server %s request failed for %s (attempt %d/%d): %s; retrying after %.1fs",
                endpoint,
                dataset_id,
                attempt,
                DATASETS_SERVER_RETRIES,
                exc,
                backoff,
            )
            time.sleep(backoff)
    assert last_error is not None
    raise last_error


def _normalize_remote_record(record: dict[str, Any]) -> dict[str, Any]:
    payload = dict(record)
    problem = _pick_first(
        payload,
        (
            "problem",
            "question",
            "Question",
            "prompt",
            "Prompt",
            "input",
            "Input",
            "query",
            "instruction",
            "Instruction",
        ),
    )
    solution = _pick_first(
        payload,
        (
            "solution",
            "Solution",
            "response",
            "Response",
            "output",
            "Output",
            "rationale",
            "cot",
            "reasoning",
            "analysis",
            "explanation",
            "generated_solution",
            "Generated Solution",
        ),
    )
    final_answer = _pick_first(
        payload,
        (
            "final_answer",
            "Final Answer",
            "answer",
            "Answer",
            "target",
            "Target",
            "label",
            "Label",
            "expected_answer",
            "Expected Answer",
        ),
    )
    if problem:
        payload["problem"] = problem
    if solution:
        payload["solution"] = solution
    if final_answer:
        payload["final_answer"] = final_answer
    return payload


def _parse_available_configs_from_error(exc: Exception) -> list[str]:
    message = str(exc or "")
    match = AVAILABLE_CONFIGS_RE.search(message)
    if not match:
        return []
    parsed: list[str] = []
    for raw in match.group(1).split(","):
        name = raw.strip().strip("'\"")
        if name:
            parsed.append(name)
    return parsed


def _choose_training_split(
    split_names: list[str],
    *,
    requested_split: str,
) -> str:
    if requested_split:
        return requested_split if requested_split in split_names else requested_split
    for name in ("train", "default", "validation"):
        if name in split_names:
            return name
    lowered = {name.lower(): name for name in split_names}
    for name in ("cot", "tir", "genselect", "additional_problems"):
        if name in lowered:
            return lowered[name]
    for lowered_name, original_name in lowered.items():
        if any(token in lowered_name for token in ("test", "eval", "validation", "valid", "dev")):
            continue
        return original_name
    return ""


def _estimate_materialization_max_rows(
    *,
    max_samples: int,
    source_weight: float,
    processing_config: dict[str, Any],
) -> int:
    effective_weight = max(float(source_weight or 0.0), 0.1)
    multiplier = max(int(effective_weight * 10), 1)
    target_examples = max(1, ceil(max_samples / multiplier))
    max_examples_per_source = int(processing_config.get("max_examples_per_source") or 0)
    if max_examples_per_source > 0:
        target_examples = min(target_examples, max_examples_per_source)
        hard_cap = max(MATERIALIZATION_HARD_CAP, max_examples_per_source * 2)
    else:
        hard_cap = MATERIALIZATION_HARD_CAP
    row_budget = max(MATERIALIZATION_MIN_ROWS, target_examples * MATERIALIZATION_OVERSAMPLE_FACTOR)
    return min(row_budget, hard_cap)


def _materialize_via_datasets_server(
    entry: dict[str, Any],
    dataset_id: str,
    *,
    max_rows: int | None,
    base_url: str = DEFAULT_DATASETS_SERVER_BASE,
) -> list[dict[str, Any]]:
    requested_config = str(
        entry.get("config")
        or entry.get("config_name")
        or entry.get("subset")
        or entry.get("dataset_config")
        or ""
    ).strip()
    requested_split = str(entry.get("split") or "").strip()
    def _fetch_rows(
        *,
        config_name: str,
        split_name: str,
        row_budget: int | None,
    ) -> list[dict[str, Any]]:
        fetched_rows: list[dict[str, Any]] = []
        offset = 0
        page_size = 100
        while row_budget is None or offset < row_budget:
            length = page_size if row_budget is None else min(page_size, row_budget - offset)
            try:
                resp = _datasets_server_get(
                    "rows",
                    params={
                        "dataset": dataset_id,
                        "config": config_name or "default",
                        "split": split_name,
                        "offset": offset,
                        "length": length,
                    },
                    timeout=30,
                    dataset_id=dataset_id,
                    base_url=base_url,
                )
                payload = resp.json()
            except Exception as exc:
                if fetched_rows:
                    LOGGER.info(
                        "datasets-server rows pagination stopped early for %s at offset %d after %d rows: %s",
                        dataset_id,
                        offset,
                        len(fetched_rows),
                        exc,
                    )
                break

            server_rows = payload.get("rows")
            if not isinstance(server_rows, list) or not server_rows:
                break

            added = 0
            for item in server_rows:
                if not isinstance(item, dict):
                    continue
                row = item.get("row")
                if not isinstance(row, dict):
                    continue
                normalized = _normalize_remote_record(row)
                if config_name:
                    normalized.setdefault("dataset_config", config_name)
                fetched_rows.append(normalized)
                added += 1
                if row_budget is not None and len(fetched_rows) >= row_budget:
                    return fetched_rows
            if added == 0:
                break
            offset += added
        return fetched_rows

    def _direct_rows_fallback() -> list[dict[str, Any]]:
        if requested_split:
            split_candidates = [requested_split]
        else:
            split_candidates = [
                "train",
                "default",
                "validation",
                "cot",
                "tir",
                "genselect",
                "additional_problems",
            ]
        if requested_config:
            config_candidates = [requested_config]
        else:
            config_candidates = ["default", ""]

        for config_name in config_candidates:
            for split_name in split_candidates:
                if any(token in split_name.lower() for token in ("test", "eval", "dev")):
                    continue
                rows = _fetch_rows(
                    config_name=config_name,
                    split_name=split_name,
                    row_budget=max_rows,
                )
                if rows:
                    LOGGER.info(
                        "Materialized %d rows for %s via datasets-server direct rows fallback "
                        "(config=%s, split=%s)",
                        len(rows),
                        dataset_id,
                        config_name or "default",
                        split_name,
                    )
                    return rows
        return []

    try:
        splits_resp = _datasets_server_get(
            "splits",
            params={"dataset": dataset_id},
            timeout=20,
            dataset_id=dataset_id,
            base_url=base_url,
        )
        split_payload = splits_resp.json()
    except Exception:
        LOGGER.info(
            "datasets-server splits lookup failed for %s; trying direct rows fallback",
            dataset_id,
        )
        return _direct_rows_fallback()

    split_entries = split_payload.get("splits")
    if not isinstance(split_entries, list) or not split_entries:
        return _direct_rows_fallback()

    grouped: dict[str, list[str]] = defaultdict(list)
    for item in split_entries:
        if not isinstance(item, dict):
            continue
        config_name = str(item.get("config") or "").strip()
        split_name = str(item.get("split") or "").strip()
        if not split_name:
            continue
        grouped[config_name].append(split_name)

    if requested_config:
        configs_to_try = [requested_config]
    else:
        configs_to_try = [cfg for cfg in grouped.keys() if cfg] or [""]

    if max_rows is None:
        per_config_budget = None
    elif len(configs_to_try) > 1:
        per_config_budget = max(1, ceil(max_rows / len(configs_to_try)))
    else:
        per_config_budget = max_rows

    rows: list[dict[str, Any]] = []
    for config_name in configs_to_try:
        if max_rows is None:
            remaining = None
        else:
            remaining = max_rows - len(rows)
            if remaining <= 0:
                break
        split_names = grouped.get(config_name, [])
        chosen_split = _choose_training_split(split_names, requested_split=requested_split)
        if not chosen_split:
            continue
        target = per_config_budget if remaining is None else min(remaining, per_config_budget)
        loaded = _fetch_rows(
            config_name=config_name,
            split_name=chosen_split,
            row_budget=target,
        )
        rows.extend(loaded)
        if max_rows is not None and len(rows) >= max_rows:
            return rows

    return rows


def _load_via_datasets_library(
    entry: dict[str, Any],
    dataset_id: str,
    *,
    max_rows: int | None,
    get_dataset_config_names,
    get_dataset_split_names,
    load_dataset,
    hf_endpoint: str | None = None,
) -> list[dict[str, Any]]:
    """
    Load dataset using datasets.load_dataset library.
    This automatically uses HF_ENDPOINT mirror if configured.
    """
    # Configure HuggingFace environment variables to use mirror
    # HF_HUB_URL takes precedence over HF_ENDPOINT in newer versions
    hf_mirror = str(hf_endpoint or os.getenv("HF_ENDPOINT", DEFAULT_HF_ENDPOINT)).rstrip("/")
    os.environ["HF_HUB_URL"] = hf_mirror
    os.environ["HF_ENDPOINT"] = hf_mirror

    # Increase timeout to avoid network issues (default is 10s, too short for China)
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"  # 5 minutes

    # Enable offline mode if network is problematic (will use cache)
    # This can be overridden by setting HF_DATASETS_OFFLINE=0
    if os.getenv("HF_DATASETS_OFFLINE") is None and not os.getenv("FORCE_ONLINE"):
        # Check if we have cache available
        cache_dir = Path(os.getenv("HF_DATASETS_CACHE", Path.home() / ".cache" / "huggingface" / "datasets"))
        dataset_cache = cache_dir / dataset_id.replace("/", "___")
        if dataset_cache.exists():
            os.environ["HF_DATASETS_OFFLINE"] = "1"
            LOGGER.info("Using offline mode for %s (cache exists at %s)", dataset_id, dataset_cache)

    requested_config = str(
        entry.get("config")
        or entry.get("config_name")
        or entry.get("subset")
        or entry.get("dataset_config")
        or ""
    ).strip()

    config_names: list[str] = []
    try:
        config_names = [str(name) for name in (get_dataset_config_names(dataset_id) or []) if str(name).strip()]
    except Exception:
        config_names = []

    if requested_config:
        configs_to_try = [requested_config]
    elif config_names:
        configs_to_try = config_names
    else:
        configs_to_try = [""]

    def _split_names_for(config_name: str) -> list[str]:
        try:
            kwargs = {"path": dataset_id}
            if config_name:
                kwargs["config_name"] = config_name
            return [str(name) for name in (get_dataset_split_names(**kwargs) or []) if str(name).strip()]
        except Exception:
            return []

    def _load_rows_for(
        config_name: str,
        chosen_split: str,
        row_budget: int | None,
    ) -> tuple[list[dict[str, Any]], Exception | None]:
        load_args: list[str] = [dataset_id]
        if config_name:
            load_args.append(config_name)
        slice_spec = chosen_split if row_budget is None else f"{chosen_split}[:{row_budget}]"
        try:
            dataset = load_dataset(*load_args, split=slice_spec)
        except Exception as exc:
            if row_budget is not None:
                try:
                    dataset = load_dataset(*load_args, split=chosen_split)
                except Exception:
                    return [], exc
            else:
                return [], exc

        loaded_rows: list[dict[str, Any]] = []
        for idx, row in enumerate(dataset):
            if row_budget is not None and idx >= row_budget:
                break
            if isinstance(row, dict):
                normalized = _normalize_remote_record(row)
                if config_name:
                    normalized.setdefault("dataset_config", config_name)
                loaded_rows.append(normalized)
        return loaded_rows, None

    requested_split = str(entry.get("split") or "").strip()

    rows = []
    if max_rows is None:
        per_config_budget = None
    elif len(configs_to_try) > 1:
        per_config_budget = max(1, ceil(max_rows / len(configs_to_try)))
    else:
        per_config_budget = max_rows

    for config_name in list(configs_to_try):
        if max_rows is None:
            remaining = None
            row_budget = per_config_budget
        else:
            remaining = max_rows - len(rows)
            if remaining <= 0:
                break
            row_budget = min(remaining, per_config_budget)
        split_names = _split_names_for(config_name)
        if split_names:
            chosen_split = _choose_training_split(split_names, requested_split=requested_split)
            if not chosen_split:
                continue
        else:
            chosen_split = requested_split or "train"
        loaded_rows, load_error = _load_rows_for(config_name, chosen_split, row_budget)
        if load_error is not None and not config_name:
            inferred_configs = _parse_available_configs_from_error(load_error)
            if inferred_configs:
                retry_budget = None if max_rows is None else (max_rows - len(rows))
                retry_per_config = None if retry_budget is None else max(1, ceil(retry_budget / len(inferred_configs)))
                for inferred_config in inferred_configs:
                    inferred_remaining = None if max_rows is None else (max_rows - len(rows))
                    if inferred_remaining is not None and inferred_remaining <= 0:
                        break
                    inferred_split_names = _split_names_for(inferred_config)
                    if inferred_split_names:
                        inferred_split = _choose_training_split(
                            inferred_split_names,
                            requested_split=requested_split,
                        )
                        if not inferred_split:
                            continue
                    else:
                        inferred_split = requested_split or "train"
                    inferred_budget = None if inferred_remaining is None else min(inferred_remaining, retry_per_config)
                    inferred_rows, _ = _load_rows_for(
                        inferred_config,
                        inferred_split,
                        inferred_budget,
                    )
                    rows.extend(inferred_rows)
                continue
        rows.extend(loaded_rows)

    return rows


def _materialized_path_has_rows(path: Path) -> bool:
    if not path.exists() or path.stat().st_size <= 0:
        return False
    suffix = path.suffix.lower()
    try:
        if suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        return True
            return False
        if suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                return len(payload) > 0
            if isinstance(payload, dict):
                for key in ("data", "examples", "rows", "items"):
                    value = payload.get(key)
                    if isinstance(value, list) and value:
                        return True
                return bool(payload)
            return False
        return True
    except Exception:
        return False


def materialize_dataset_entry(
    entry: dict[str, Any],
    cache_dir: str | Path,
    *,
    max_rows: int | None = 2048,
    data_access_config: dict[str, Any] | None = None,
) -> str:
    full_local_path = _get_entry_full_local_path(entry)
    if full_local_path:
        local_path_obj = Path(full_local_path)
        if _materialized_path_has_rows(local_path_obj):
            return full_local_path
        LOGGER.warning(
            "Ignoring stale or empty materialized dataset path for %s: %s",
            entry.get("source_id") or entry.get("name") or "unknown",
            local_path_obj,
        )

    dataset_id = _extract_hf_dataset_id(entry)
    if not dataset_id:
        return ""

    # ── 优先通过 sandbox 代理物化 ──
    if _HF_SANDBOX_URL:
        sandbox_path = _sandbox_materialize(entry, max_rows)
        if sandbox_path:
            _set_entry_full_local_path(entry, sandbox_path)
            return sandbox_path

    # Use global shared cache if MATH_PT_SHARED_CACHE is set
    shared_cache_root = os.getenv("MATH_PT_SHARED_CACHE")
    if shared_cache_root:
        cache_dir = Path(shared_cache_root) / "materialized_datasets"
        LOGGER.info("Using shared cache directory: %s", cache_dir)
    else:
        cache_dir = Path(cache_dir)

    cache_dir.mkdir(parents=True, exist_ok=True)

    # Include config in filename to avoid collisions
    config_name = str(
        entry.get("config")
        or entry.get("config_name")
        or entry.get("subset")
        or entry.get("dataset_config")
        or "default"
    ).strip()

    max_rows_tag = "all" if max_rows is None else str(max_rows)
    cache_key = f"{_safe_source_filename(dataset_id)}_{config_name}_{max_rows_tag}"
    output_path = cache_dir / f"{cache_key}.jsonl"

    if output_path.exists():
        if _materialized_path_has_rows(output_path):
            LOGGER.info("Found cached materialized dataset at: %s", output_path)
            return str(output_path)
        LOGGER.warning("Found empty cached materialized dataset at %s; removing and rematerializing", output_path)
        output_path.unlink(missing_ok=True)

    data_access = normalize_data_access_config(data_access_config)
    hf_endpoint = str(data_access.get("hf_endpoint") or DEFAULT_HF_ENDPOINT)
    datasets_server_cfg = dict(data_access.get("datasets_server") or {})
    datasets_server_enabled = bool(datasets_server_cfg.get("enabled", False))
    datasets_server_base = str(datasets_server_cfg.get("base_url") or DEFAULT_DATASETS_SERVER_BASE)

    rows: list[dict[str, Any]] = []

    # Strategy: Prefer load_dataset (faster, uses HF_ENDPOINT mirror, no rate limits)
    # Only fall back to datasets-server if load_dataset fails
    prefer_direct_load = not _env_flag("MATH_PT_FORCE_DATASETS_SERVER")

    if prefer_direct_load:
        # Try load_dataset first (uses HF mirror if HF_ENDPOINT is set)
        LOGGER.info(
            "Attempting direct load_dataset for %s (uses HF_ENDPOINT=%s if set)",
            dataset_id,
            os.getenv("HF_ENDPOINT", "default"),
        )
        from datasets import get_dataset_config_names, get_dataset_split_names, load_dataset

        try:
            rows = _load_via_datasets_library(
                entry=entry,
                dataset_id=dataset_id,
                max_rows=max_rows,
                get_dataset_config_names=get_dataset_config_names,
                get_dataset_split_names=get_dataset_split_names,
                load_dataset=load_dataset,
                hf_endpoint=hf_endpoint,
            )
        except Exception as e:
            LOGGER.info(
                "Direct load_dataset failed for %s: %s; falling back to datasets-server",
                dataset_id,
                e,
            )
            rows = []

    # Fall back to datasets-server if needed
    if not rows and datasets_server_enabled and not _env_flag("MATH_PT_DISABLE_DATASETS_SERVER"):
        rows = _materialize_via_datasets_server(
            entry,
            dataset_id,
            max_rows=max_rows,
            base_url=datasets_server_base,
        )
    elif not rows and not datasets_server_enabled:
        LOGGER.info(
            "Skipping datasets-server fallback for %s because data_access.datasets_server.enabled=false",
            dataset_id,
        )

    if rows:
        write_jsonl(output_path, rows)
        return str(output_path)

    # Final fallback: try load_dataset one more time
    LOGGER.info(
        "All methods failed for %s; attempting final fallback with load_dataset",
        dataset_id,
    )

    from datasets import get_dataset_config_names, get_dataset_split_names, load_dataset

    requested_config = str(
        entry.get("config")
        or entry.get("config_name")
        or entry.get("subset")
        or entry.get("dataset_config")
        or ""
    ).strip()

    config_names: list[str] = []
    try:
        config_names = [str(name) for name in (get_dataset_config_names(dataset_id) or []) if str(name).strip()]
    except Exception:
        config_names = []

    if requested_config:
        configs_to_try = [requested_config]
    elif config_names:
        configs_to_try = config_names
    else:
        configs_to_try = [""]

    def _split_names_for(config_name: str) -> list[str]:
        try:
            kwargs = {"path": dataset_id}
            if config_name:
                kwargs["config_name"] = config_name
            return [str(name) for name in (get_dataset_split_names(**kwargs) or []) if str(name).strip()]
        except Exception:
            return []

    def _load_rows_for(
        config_name: str,
        chosen_split: str,
        row_budget: int | None,
    ) -> tuple[list[dict[str, Any]], Exception | None]:
        load_args: list[str] = [dataset_id]
        if config_name:
            load_args.append(config_name)
        slice_spec = chosen_split if row_budget is None else f"{chosen_split}[:{row_budget}]"
        try:
            dataset = load_dataset(*load_args, split=slice_spec)
        except Exception as exc:
            if row_budget is not None:
                try:
                    dataset = load_dataset(*load_args, split=chosen_split)
                except Exception:
                    return [], exc
            else:
                return [], exc

        loaded_rows: list[dict[str, Any]] = []
        for idx, row in enumerate(dataset):
            if row_budget is not None and idx >= row_budget:
                break
            if isinstance(row, dict):
                normalized = _normalize_remote_record(row)
                if config_name:
                    normalized.setdefault("dataset_config", config_name)
                loaded_rows.append(normalized)
        return loaded_rows, None

    requested_split = str(entry.get("split") or "").strip()

    rows = []
    if max_rows is None:
        per_config_budget = None
    elif len(configs_to_try) > 1:
        per_config_budget = max(1, ceil(max_rows / len(configs_to_try)))
    else:
        per_config_budget = max_rows

    for config_name in list(configs_to_try):
        if max_rows is None:
            remaining = None
            row_budget = per_config_budget
        else:
            remaining = max_rows - len(rows)
            if remaining <= 0:
                break
            row_budget = min(remaining, per_config_budget)
        split_names = _split_names_for(config_name)
        if split_names:
            chosen_split = _choose_training_split(split_names, requested_split=requested_split)
            if not chosen_split:
                continue
        else:
            chosen_split = requested_split or "train"
        loaded_rows, load_error = _load_rows_for(config_name, chosen_split, row_budget)
        if load_error is not None and not config_name:
            inferred_configs = _parse_available_configs_from_error(load_error)
            if inferred_configs:
                retry_budget = None if max_rows is None else (max_rows - len(rows))
                retry_per_config = None if retry_budget is None else max(1, ceil(retry_budget / len(inferred_configs)))
                for inferred_config in inferred_configs:
                    inferred_remaining = None if max_rows is None else (max_rows - len(rows))
                    if inferred_remaining is not None and inferred_remaining <= 0:
                        break
                    inferred_split_names = _split_names_for(inferred_config)
                    if inferred_split_names:
                        inferred_split = _choose_training_split(
                            inferred_split_names,
                            requested_split=requested_split,
                        )
                        if not inferred_split:
                            continue
                    else:
                        inferred_split = requested_split or "train"
                    inferred_budget = None if inferred_remaining is None else min(inferred_remaining, retry_per_config)
                    inferred_rows, _ = _load_rows_for(
                        inferred_config,
                        inferred_split,
                        inferred_budget,
                    )
                    rows.extend(inferred_rows)
                continue
        rows.extend(loaded_rows)

    write_jsonl(output_path, rows)
    return str(output_path)


def _iter_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return read_jsonl(path)
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [row for row in payload if isinstance(row, dict)]
        if isinstance(payload, dict):
            for key in ("data", "examples", "rows", "items"):
                value = payload.get(key)
                if isinstance(value, list):
                    return [row for row in value if isinstance(row, dict)]
        return []
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    return []


def _iter_rows_limited(path: Path, limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return []

    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except Exception:
                    continue
                if isinstance(payload, dict):
                    rows.append(payload)
                if len(rows) >= limit:
                    break
        return rows
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows: list[dict[str, Any]] = []
            for row in csv.DictReader(handle):
                rows.append(dict(row))
                if len(rows) >= limit:
                    break
            return rows
    return _iter_rows(path)[:limit]


def _clip_probe_value(value: Any, *, depth: int = 0) -> Any:
    payload = make_json_serializable(value)
    if depth >= 2:
        if isinstance(payload, (dict, list)):
            return f"<{type(payload).__name__}>"
        return payload

    if isinstance(payload, str):
        text = payload.strip()
        if len(text) <= 240:
            return text
        return text[:237].rstrip() + "..."
    if isinstance(payload, dict):
        summary: dict[str, Any] = {}
        for idx, (key, item) in enumerate(payload.items()):
            if idx >= 16:
                summary["__truncated__"] = f"+{len(payload) - idx} fields"
                break
            summary[str(key)] = _clip_probe_value(item, depth=depth + 1)
        return summary
    if isinstance(payload, list):
        clipped = [_clip_probe_value(item, depth=depth + 1) for item in payload[:8]]
        if len(payload) > 8:
            clipped.append(f"... (+{len(payload) - 8} more)")
        return clipped
    return payload


def _summarize_probe_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys: Counter[str] = Counter()
    for row in rows:
        for key in row.keys():
            keys[str(key)] += 1
    return {
        "sample_row_count": len(rows),
        "schema_keys": sorted(keys.keys()),
        "field_presence": dict(sorted(keys.items())),
        "preview_rows": [_clip_probe_value(row) for row in rows[:DEFAULT_PROBE_PREVIEW_ROWS]],
    }


def infer_coverage_tags(path: Path, rows: list[dict[str, Any]]) -> list[str]:
    tags: set[str] = {"math_reasoning"}
    name = path.stem.lower()
    if "aime" in name:
        tags.add("aime")
    if "gsm" in name:
        tags.add("gsm8k")
    if "math" in name:
        tags.add("competition_math")
    if any("geometry" in json_dumps_safe(row, ensure_ascii=False).lower() for row in rows[:20]):
        tags.add("geometry")
    if any("mod" in json_dumps_safe(row, ensure_ascii=False).lower() for row in rows[:20]):
        tags.add("number_theory")
    return sorted(tags)


def _default_instruction_for_task(task_type: str) -> str:
    if _is_tool_using_task(task_type):
        return DEFAULT_TOOL_USE_INSTRUCTION
    if _is_code_generation_task(task_type):
        return DEFAULT_CODE_INSTRUCTION
    return DEFAULT_INSTRUCTION


def _is_tool_using_task(task_type: str | None) -> bool:
    return _normalize_label(task_type or "") in TOOL_USE_TASK_TYPES


def _is_code_generation_task(task_type: str | None) -> bool:
    return _normalize_label(task_type or "") in CODE_GENERATION_TASK_TYPES


def _infer_task_type(entry: dict[str, Any], task_type: str | None) -> str:
    normalized = _normalize_label(task_type or "")
    if normalized and normalized != "math_reasoning":
        return task_type or "math_reasoning"

    hints: list[str] = []
    for key in ("source_id", "name", "url"):
        value = entry.get(key)
        if value:
            hints.append(str(value))
    for tag in entry.get("coverage_tags") or []:
        if tag:
            hints.append(str(tag))
    blob = " ".join(hints).lower()
    if any(token in blob for token in ("mbpp", "human_eval", "humaneval", "code generation", "code_generation", "python problem", "coding")):
        return "code_generation"
    if any(token in blob for token in ("function-calling", "function_calling", "tool calling", "tool_calling", "tool-use", "tool_use", "api calling", "api_calling", "xlam", "glaive", "hermes")):
        return "function_calling"
    return task_type or "math_reasoning"


def _load_python_module_from_path(path: str | Path, prefix: str):
    module_path = Path(path)
    module_name = f"{prefix}_{hashlib.md5(str(module_path).encode('utf-8')).hexdigest()}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load python module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_custom_data_adapter(script_path: str | Path | None) -> Any | None:
    if not script_path:
        return None
    path = Path(script_path)
    if not path.exists():
        return None

    module = _load_python_module_from_path(path, "_math_pt_adapter")
    adapter_obj = getattr(module, "adapter", None)
    if adapter_obj is not None:
        return adapter_obj

    adapter_cls = getattr(module, "DataAdapter", None)
    if callable(adapter_cls):
        return adapter_cls()

    build_fn = getattr(module, "build_adapter", None)
    if callable(build_fn):
        return build_fn()

    raise ValueError(
        f"adapter script {path} must define `adapter`, `DataAdapter`, or callable `build_adapter()`"
    )


def _iter_selected_dataset_entries(
    dataset_entries: list[dict[str, Any]],
    processing_config: dict[str, Any],
    source_weights: dict[str, float] | None,
) -> list[tuple[dict[str, Any], str, str, float]]:
    source_weights = source_weights or {}
    include_sources = {
        str(item).strip()
        for item in (processing_config.get("include_sources") or [])
        if str(item).strip()
    }
    exclude_sources = {
        str(item).strip()
        for item in (processing_config.get("exclude_sources") or [])
        if str(item).strip()
    }

    selected: list[tuple[dict[str, Any], str, str, float]] = []
    for entry in dataset_entries:
        source_id = str(entry.get("source_id") or entry.get("name") or "unknown_source")
        task_type = _infer_task_type(entry, entry.get("task_type"))
        if include_sources and source_id not in include_sources:
            continue
        if source_id in exclude_sources:
            continue
        raw_weight = float(source_weights.get(source_id, 1.0) or 0.0)
        if raw_weight <= 0:
            continue
        selected.append((entry, source_id, task_type, raw_weight))
    return selected


def _strip_tool_chat_tokens(text: str) -> str:
    return str(text or "").replace("<|endoftext|>", "").strip()


def _normalize_dialog_role(role: str | None) -> str:
    normalized = _normalize_label(role or "")
    if normalized in {"human", "user"}:
        return "USER"
    if normalized in {"assistant", "gpt", "model"}:
        return "ASSISTANT"
    if normalized in {"function_response", "function", "tool", "tool_response", "observation"}:
        return "FUNCTION RESPONSE"
    if normalized == "system":
        return "SYSTEM"
    return (role or "UNKNOWN").strip().upper() or "UNKNOWN"


def _render_dialog_messages(messages: list[tuple[str, str]]) -> str:
    rendered: list[str] = []
    for role, content in messages:
        cleaned = _strip_tool_chat_tokens(content)
        if not cleaned:
            continue
        rendered.append(f"{role}: {cleaned}")
    return "\n\n".join(rendered).strip()


def _build_tool_using_example(
    *,
    source_id: str,
    example_id: str,
    task_type: str,
    instruction: str,
    messages: list[tuple[str, str]],
    difficulty: str = "unknown",
    metadata: dict[str, Any] | None = None,
) -> MathTrainingExample | None:
    cleaned_messages = [(role, _strip_tool_chat_tokens(content)) for role, content in messages]
    cleaned_messages = [(role, content) for role, content in cleaned_messages if content]
    if not cleaned_messages:
        return None

    last_assistant_idx = None
    for idx in range(len(cleaned_messages) - 1, -1, -1):
        if cleaned_messages[idx][0] == "ASSISTANT":
            last_assistant_idx = idx
            break
    if last_assistant_idx is None or last_assistant_idx == 0:
        return None

    prompt = _render_dialog_messages(cleaned_messages[:last_assistant_idx])
    output = cleaned_messages[last_assistant_idx][1].strip()
    if not prompt or not output:
        return None

    return MathTrainingExample(
        example_id=example_id,
        source=source_id,
        problem=prompt,
        solution=output,
        final_answer=normalize_text(output),
        answer_style="tool_using",
        difficulty=difficulty,
        task_type=task_type,
        instruction=instruction.strip() or _default_instruction_for_task(task_type),
        metadata=dict(metadata or {}),
    )


def _load_glaive_tool_examples(path: str | Path, source_id: str, task_type: str) -> list[MathTrainingExample]:
    file_path = Path(path)
    rows = _iter_rows(file_path)
    examples: list[MathTrainingExample] = []
    pattern = re.compile(r"(?m)^(USER|ASSISTANT|FUNCTION RESPONSE):\s*")
    for idx, row in enumerate(rows):
        chat = str(row.get("chat") or "")
        matches = list(pattern.finditer(chat))
        messages: list[tuple[str, str]] = []
        for match_index, match in enumerate(matches):
            start = match.end()
            end = matches[match_index + 1].start() if match_index + 1 < len(matches) else len(chat)
            content = _strip_tool_chat_tokens(chat[start:end])
            if content:
                messages.append((match.group(1), content))
        instruction = _strip_tool_chat_tokens(str(row.get("system") or ""))
        metadata = {
            "raw_path": str(file_path),
            "topic": row.get("dataset_config") or row.get("domain") or "tool_using",
        }
        example = _build_tool_using_example(
            source_id=source_id,
            example_id=f"{source_id}:{idx}",
            task_type=task_type,
            instruction=instruction,
            messages=messages,
            metadata=metadata,
        )
        if example is not None:
            examples.append(example)
    return examples


def _load_hermes_tool_examples(path: str | Path, source_id: str, task_type: str) -> list[MathTrainingExample]:
    file_path = Path(path)
    rows = _iter_rows(file_path)
    examples: list[MathTrainingExample] = []
    for idx, row in enumerate(rows):
        conversations = row.get("conversations") or []
        if not isinstance(conversations, list):
            continue
        instruction_parts: list[str] = []
        messages: list[tuple[str, str]] = []
        for item in conversations:
            if not isinstance(item, dict):
                continue
            role = _normalize_dialog_role(item.get("from"))
            value = _strip_tool_chat_tokens(str(item.get("value") or ""))
            if not value:
                continue
            if role == "SYSTEM":
                instruction_parts.append(value)
            else:
                messages.append((role, value))
        tools = _strip_tool_chat_tokens(str(row.get("tools") or ""))
        instruction = "\n\n".join(part for part in instruction_parts if part).strip()
        if tools and tools not in instruction:
            if "<tools>" not in instruction:
                instruction = f"{instruction}\n\n<tools>\n{tools}\n</tools>".strip()
        topic_parts = [row.get("category"), row.get("subcategory"), row.get("task")]
        metadata = {
            "raw_path": str(file_path),
            "topic": " / ".join(str(part).strip() for part in topic_parts if str(part).strip()) or "tool_using",
            "category": row.get("category") or "",
            "subcategory": row.get("subcategory") or "",
            "task": row.get("task") or "",
        }
        example = _build_tool_using_example(
            source_id=source_id,
            example_id=str(row.get("id") or f"{source_id}:{idx}"),
            task_type=task_type,
            instruction=instruction,
            messages=messages,
            metadata=metadata,
        )
        if example is not None:
            examples.append(example)
    return examples


def load_tool_using_examples(path: str | Path, source_id: str, task_type: str) -> list[MathTrainingExample]:
    file_path = Path(path)
    rows = _iter_rows(file_path)
    if not rows:
        return []
    sample = rows[0]
    if isinstance(sample, dict) and "conversations" in sample:
        return _load_hermes_tool_examples(file_path, source_id=source_id, task_type=task_type)
    if isinstance(sample, dict) and "chat" in sample:
        return _load_glaive_tool_examples(file_path, source_id=source_id, task_type=task_type)
    return []


def load_code_generation_examples(path: str | Path, source_id: str, task_type: str) -> list[MathTrainingExample]:
    file_path = Path(path)
    rows = _iter_rows(file_path)
    examples: list[MathTrainingExample] = []
    for idx, row in enumerate(rows):
        problem = _pick_first(row, ("problem", "question", "text", "prompt", "input"))
        solution = _pick_first(
            row,
            ("solution", "code", "canonical_solution", "response", "output", "completion", "answer"),
        )
        if not problem or not solution:
            continue

        example_id = str(
            row.get("task_id")
            or row.get("id")
            or row.get("entry_point")
            or f"{source_id}:{idx}"
        )
        metadata = {
            "raw_path": str(file_path),
            "topic": row.get("topic") or row.get("domain") or "code_generation",
        }
        for key in ("task_id", "entry_point", "test", "test_list", "test_setup_code", "challenge_test_list"):
            value = row.get(key)
            if value not in (None, "", []):
                metadata[key] = value

        examples.append(
            MathTrainingExample(
                example_id=example_id,
                source=source_id,
                problem=normalize_text(problem),
                solution=solution.strip(),
                final_answer=_normalize_code_answer(solution),
                answer_style="code",
                difficulty=str(row.get("difficulty", "unknown") or "unknown"),
                task_type=task_type,
                instruction=_default_instruction_for_task(task_type),
                metadata=metadata,
            )
        )
    return examples


def load_task_aligned_examples(path: str | Path, source_id: str, task_type: str) -> list[MathTrainingExample]:
    if _is_code_generation_task(task_type):
        return load_code_generation_examples(path, source_id=source_id, task_type=task_type)
    if _is_tool_using_task(task_type):
        return load_tool_using_examples(path, source_id=source_id, task_type=task_type)
    return load_math_examples(path, source_id=source_id)


def load_math_examples(path: str | Path, source_id: str) -> list[MathTrainingExample]:
    file_path = Path(path)
    rows = _iter_rows(file_path)
    examples: list[MathTrainingExample] = []
    for idx, row in enumerate(rows):
        problem = _pick_first(row, ("problem", "question", "prompt", "input"))
        solution = _pick_first(row, ("solution", "response", "output", "rationale", "cot"))
        final_answer = _pick_first(row, ("final_answer", "answer", "target", "label"))
        if not final_answer:
            final_answer = normalize_final_answer(solution)
        else:
            final_answer = normalize_final_answer(final_answer)
        if not problem or not final_answer:
            continue
        solution_text = solution or f"ANSWER: {final_answer}"
        answer_style = _derive_answer_style(solution_text)
        examples.append(
            MathTrainingExample(
                example_id=f"{source_id}:{idx}",
                source=source_id,
                problem=normalize_text(problem),
                solution=solution_text.strip(),
                final_answer=final_answer,
                answer_style=answer_style,
                difficulty=str(row.get("difficulty", "unknown") or "unknown"),
                instruction=DEFAULT_INSTRUCTION,
                metadata={
                    "raw_path": str(file_path),
                    "topic": row.get("topic") or row.get("domain") or "unknown",
                },
            )
        )
    return examples


def _coerce_training_example(
    result: dict[str, Any],
    *,
    fallback: MathTrainingExample | None,
    source_id: str,
    task_type: str,
    example_id: str,
    default_metadata: dict[str, Any] | None = None,
) -> MathTrainingExample | None:
    effective_task_type = str(result.get("task_type") or (fallback.task_type if fallback else task_type) or "math_reasoning")
    effective_source = str(result.get("source") or (fallback.source if fallback else source_id))
    effective_example_id = str(result.get("example_id") or (fallback.example_id if fallback else example_id))
    problem = normalize_text(str(result.get("problem") or (fallback.problem if fallback else "") or ""))
    solution = str(result.get("solution") or (fallback.solution if fallback else "") or "").strip()
    raw_final_answer = str(result.get("final_answer") or "")

    if _is_tool_using_task(effective_task_type):
        final_answer = normalize_final_answer(raw_final_answer)
        if not final_answer:
            final_answer = normalize_text(solution) or (fallback.final_answer if fallback else "")
        if not problem or not solution:
            return None
    elif _is_code_generation_task(effective_task_type):
        final_answer = _normalize_code_answer(raw_final_answer or solution or (fallback.final_answer if fallback else ""))
        if not final_answer or not problem or not solution:
            return None
    else:
        final_answer = normalize_final_answer(raw_final_answer)
        if not final_answer:
            final_answer = normalize_final_answer(solution) or (fallback.final_answer if fallback else "")
        if not problem or not final_answer:
            return None

    raw_answer_style = result.get("answer_style")
    answer_style = _normalize_label(raw_answer_style or "")
    solution_changed = fallback is not None and solution != (fallback.solution or "").strip()
    if _is_tool_using_task(effective_task_type):
        if raw_answer_style is None:
            answer_style = "tool_using"
    elif _is_code_generation_task(effective_task_type):
        if raw_answer_style is None or answer_style in {"", "short_answer", "long_reasoning"}:
            answer_style = "code"
    elif (
        raw_answer_style is None
        or answer_style not in {"short_answer", "long_reasoning"}
        or (solution_changed and fallback is not None and answer_style == fallback.answer_style)
    ):
        answer_style = _derive_answer_style(solution or f"ANSWER: {final_answer}")

    metadata = result.get("metadata")
    if not isinstance(metadata, dict):
        metadata = dict(fallback.metadata) if fallback is not None else dict(default_metadata or {})
    instruction = str(
        result.get("instruction")
        or (fallback.instruction if fallback is not None else "")
        or _default_instruction_for_task(effective_task_type)
    )

    return MathTrainingExample(
        example_id=effective_example_id,
        source=effective_source,
        problem=problem,
        solution=solution or f"ANSWER: {final_answer}",
        final_answer=final_answer,
        answer_style=answer_style,
        difficulty=str(result.get("difficulty") or (fallback.difficulty if fallback is not None else "unknown") or "unknown"),
        task_type=effective_task_type,
        instruction=instruction,
        metadata=metadata,
    )


def _coerce_validation_result(value: Any) -> tuple[bool, str]:
    if isinstance(value, tuple) and len(value) >= 2:
        return bool(value[0]), str(value[1] or "")
    if isinstance(value, bool):
        return value, "" if value else "validate_example returned False"
    return bool(value), "" if value else "validate_example returned a falsy value"


def _adapter_supports_meta(fn: Any) -> bool:
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    params = list(signature.parameters.values())
    if any(param.kind == inspect.Parameter.VAR_POSITIONAL for param in params):
        return True
    positional = [
        param
        for param in params
        if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    return len(positional) >= 2


def _call_adapter_hook(fn: Any, payload: dict[str, Any], meta: dict[str, Any]) -> Any:
    if _adapter_supports_meta(fn):
        return fn(dict(payload), dict(meta))
    return fn(dict(payload))


def _run_custom_data_adapter_on_rows(
    rows: list[dict[str, Any]],
    *,
    source_id: str,
    task_type: str,
    raw_path: str,
    adapter_script_path: str | Path,
) -> tuple[list[MathTrainingExample], dict[str, Any]]:
    adapter = _load_custom_data_adapter(adapter_script_path)
    if adapter is None:
        raise ValueError(f"custom adapter path not found: {adapter_script_path}")

    adapt_fn = getattr(adapter, "adapt", None)
    to_example_fn = getattr(adapter, "to_example", None)
    if not callable(adapt_fn) and not callable(to_example_fn):
        raise ValueError(f"adapter {adapter_script_path} must define callable adapt(row[, meta]) or to_example(row[, meta])")

    inspect_fn = getattr(adapter, "inspect_raw_row", None)
    raw_transform_fn = getattr(adapter, "transform_raw_row", None)
    validate_fn = getattr(adapter, "validate_example", None)
    example_transform_fn = getattr(adapter, "transform_example", None)

    examples: list[MathTrainingExample] = []
    drop_reasons: Counter[str] = Counter()
    preview_examples: list[dict[str, Any]] = []
    preview_failures: list[dict[str, Any]] = []
    inspected_rows = 0
    transformed_rows = 0
    example_rows = 0
    validated_rows = 0

    default_metadata = {"raw_path": raw_path, "topic": "unknown", "manifest_source_id": source_id}

    for idx, row in enumerate(rows):
        try:
            row_payload = dict(row)
        except Exception:
            drop_reasons["invalid_raw_row"] += 1
            continue

        row_meta = {
            "manifest_source_id": source_id,
            "task_type": task_type,
            "raw_path": raw_path,
            "row_index": idx,
        }
        row_payload.setdefault("__meta__", dict(row_meta))

        if callable(inspect_fn):
            try:
                _call_adapter_hook(inspect_fn, row_payload, row_meta)
            except Exception:
                pass
        inspected_rows += 1

        try:
            transformed_row = dict(row_payload)
            if callable(raw_transform_fn):
                transformed_row = _call_adapter_hook(raw_transform_fn, row_payload, row_meta)
            if transformed_row is None:
                drop_reasons["transform_raw_row:dropped"] += 1
                continue
            if not isinstance(transformed_row, dict):
                drop_reasons["transform_raw_row:invalid_type"] += 1
                continue
            transformed_rows += 1
            if isinstance(transformed_row, dict):
                transformed_row.setdefault("__meta__", dict(row_meta))

            if callable(adapt_fn):
                example_payload = _call_adapter_hook(adapt_fn, transformed_row, row_meta)
            else:
                example_payload = _call_adapter_hook(to_example_fn, transformed_row, row_meta)
            if example_payload is None:
                drop_reasons["to_example:dropped"] += 1
                continue
            if not isinstance(example_payload, dict):
                drop_reasons["to_example:invalid_type"] += 1
                continue
            example_rows += 1

            candidate = _coerce_training_example(
                example_payload,
                fallback=None,
                source_id=source_id,
                task_type=task_type,
                example_id=f"{source_id}:{idx}",
                default_metadata=default_metadata,
            )
            if candidate is None:
                drop_reasons["to_example:invalid_example"] += 1
                continue

            if callable(validate_fn):
                valid, reason = _coerce_validation_result(
                    _call_adapter_hook(validate_fn, dict(candidate.to_dict()), row_meta)
                )
                if not valid:
                    drop_reasons[f"validate_example:{reason or 'rejected'}"] += 1
                    if len(preview_failures) < 3:
                        preview_failures.append(
                            {
                                "stage": "validate_example",
                                "reason": reason or "rejected",
                                "row": _clip_probe_value(row_payload),
                            }
                        )
                    continue
            validated_rows += 1

            if callable(example_transform_fn):
                transformed_example = _call_adapter_hook(example_transform_fn, dict(candidate.to_dict()), row_meta)
                if transformed_example is None:
                    drop_reasons["transform_example:dropped"] += 1
                    continue
                if not isinstance(transformed_example, dict):
                    drop_reasons["transform_example:invalid_type"] += 1
                    continue
                candidate = _coerce_training_example(
                    transformed_example,
                    fallback=candidate,
                    source_id=source_id,
                    task_type=task_type,
                    example_id=candidate.example_id,
                    default_metadata=default_metadata,
                )
                if candidate is None:
                    drop_reasons["transform_example:invalid_example"] += 1
                    continue

            examples.append(candidate)
            if len(preview_examples) < 3:
                preview_examples.append(_clip_probe_value(candidate.to_dict()))
        except Exception as exc:
            drop_reasons[f"adapter_exception:{type(exc).__name__}"] += 1
            if len(preview_failures) < 3:
                preview_failures.append(
                    {
                        "stage": "exception",
                        "reason": f"{type(exc).__name__}: {exc}",
                        "row": _clip_probe_value(row_payload),
                    }
                )

    return examples, {
        "adapter_script_path": str(adapter_script_path),
        "raw_rows": len(rows),
        "inspected_rows": inspected_rows,
        "after_raw_transform": transformed_rows,
        "after_to_example": example_rows,
        "after_validate": validated_rows,
        "usable_examples": len(examples),
        "drop_reasons": dict(drop_reasons),
        "example_previews": preview_examples,
        "failure_previews": preview_failures,
    }


def apply_transform_script(
    examples: list[MathTrainingExample],
    script_path: str | Path | None,
) -> list[MathTrainingExample]:
    if not script_path:
        return examples
    path = Path(script_path)
    if not path.exists():
        return examples

    module = _load_python_module_from_path(path, "_math_pt_transform")
    transform_fn = getattr(module, "transform", None)
    if not callable(transform_fn):
        raise ValueError(f"transform script {path} must define callable transform(example)")

    transformed: list[MathTrainingExample] = []
    for example in examples:
        payload = example.to_dict()
        result = transform_fn(dict(payload))
        if result is None:
            continue
        if not isinstance(result, dict):
            raise TypeError(f"transform(example) must return dict | None, got {type(result)!r}")

        candidate = _coerce_training_example(
            result,
            fallback=example,
            source_id=example.source,
            task_type=example.task_type,
            example_id=example.example_id,
        )
        if candidate is not None:
            transformed.append(candidate)
    return transformed


def _apply_processing_config(
    examples: list[MathTrainingExample],
    processing_config: dict[str, Any],
) -> list[MathTrainingExample]:
    if not processing_config:
        return examples

    topic_allowlist = {
        _normalize_label(item) for item in (processing_config.get("topic_allowlist") or []) if str(item).strip()
    }
    topic_blocklist = {
        _normalize_label(item) for item in (processing_config.get("topic_blocklist") or []) if str(item).strip()
    }
    difficulty_allowlist = {
        _normalize_label(item)
        for item in (processing_config.get("difficulty_allowlist") or [])
        if str(item).strip()
    }
    difficulty_blocklist = {
        _normalize_label(item)
        for item in (processing_config.get("difficulty_blocklist") or [])
        if str(item).strip()
    }
    answer_style_allowlist = {
        _normalize_label(item)
        for item in (processing_config.get("answer_style_allowlist") or [])
        if str(item).strip()
    }
    min_problem_chars = int(processing_config.get("min_problem_chars") or 0)
    max_problem_chars = int(processing_config.get("max_problem_chars") or 0)
    min_solution_chars = int(processing_config.get("min_solution_chars") or 0)
    max_solution_chars = int(processing_config.get("max_solution_chars") or 0)
    max_examples_per_source = int(processing_config.get("max_examples_per_source") or 0)

    filtered: list[MathTrainingExample] = []
    for example in examples:
        problem_len = len(example.problem or "")
        solution_len = len(example.solution or "")
        topic = _normalize_label(example.metadata.get("topic") or "unknown")
        difficulty = _normalize_label(example.difficulty or "unknown")
        answer_style = _normalize_label(example.answer_style or "unknown")

        if topic_allowlist and topic not in topic_allowlist:
            continue
        if topic_blocklist and topic in topic_blocklist:
            continue
        if difficulty_allowlist and difficulty not in difficulty_allowlist:
            continue
        if difficulty_blocklist and difficulty in difficulty_blocklist:
            continue
        if answer_style_allowlist and answer_style not in answer_style_allowlist:
            continue
        if min_problem_chars and problem_len < min_problem_chars:
            continue
        if max_problem_chars and problem_len > max_problem_chars:
            continue
        if min_solution_chars and solution_len < min_solution_chars:
            continue
        if max_solution_chars and solution_len > max_solution_chars:
            continue
        filtered.append(example)
        if max_examples_per_source and len(filtered) >= max_examples_per_source:
            break
    return filtered


def deduplicate_examples(
    examples: list[MathTrainingExample],
    *,
    keep_mode: str = "short_and_long",
) -> list[MathTrainingExample]:
    grouped: dict[tuple[str, str], list[MathTrainingExample]] = defaultdict(list)
    for example in examples:
        if _is_tool_using_task(example.task_type) or _is_code_generation_task(example.task_type):
            key = (normalize_text(example.problem).lower(), normalize_text(example.solution).lower())
        else:
            key = (normalize_text(example.problem).lower(), normalize_final_answer(example.final_answer).lower())
        grouped[key].append(example)

    deduped: list[MathTrainingExample] = []
    for group in grouped.values():
        if any(_is_tool_using_task(item.task_type) or _is_code_generation_task(item.task_type) for item in group):
            deduped.append(group[0])
            continue
        short = next((item for item in group if item.answer_style == "short_answer"), None)
        long = next((item for item in group if item.answer_style == "long_reasoning"), None)
        if keep_mode == "short_only":
            if short:
                deduped.append(short)
            elif group:
                deduped.append(group[0])
            continue
        if keep_mode == "long_only":
            if long:
                deduped.append(long)
            elif group:
                deduped.append(group[0])
            continue
        if keep_mode == "first":
            deduped.append(group[0])
            continue
        if short:
            deduped.append(short)
        if long and long is not short:
            deduped.append(long)
        if not short and not long:
            deduped.append(group[0])
    return deduped


def _short_output(example: MathTrainingExample) -> str:
    solution = normalize_text(example.solution)
    first_sentence = re.split(r"(?<=[.!?])\s+", solution, maxsplit=1)[0]
    if normalize_final_answer(first_sentence) == example.final_answer:
        return f"ANSWER: {example.final_answer}"
    return f"{first_sentence}\n\nANSWER: {example.final_answer}"


def _long_output(example: MathTrainingExample) -> str:
    solution = example.solution.strip()
    normalized_solution = normalize_final_answer(solution)
    if normalized_solution == example.final_answer:
        return solution
    if "final answer" in solution.lower() or "ANSWER:" in solution:
        return solution
    return f"{solution}\n\nANSWER: {example.final_answer}"


def export_alpaca_dataset(
    examples: list[MathTrainingExample],
    output_path: str | Path,
    *,
    short_answer_ratio: float = 0.5,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    short_count = 0
    long_count = 0
    for index, example in enumerate(examples):
        if _is_tool_using_task(example.task_type):
            output = example.solution.strip()
            style = "tool_using"
            long_count += 1
        elif _is_code_generation_task(example.task_type):
            output = example.solution.strip()
            style = "code"
            long_count += 1
        else:
            current_ratio = short_count / max(len(rows), 1) if rows else 0.0
            prefer_short = current_ratio < short_answer_ratio
            use_short = prefer_short if example.answer_style == "short_answer" else ((index % 2) == 0 and prefer_short)
            if use_short:
                output = _short_output(example)
                style = "short_answer"
                short_count += 1
            else:
                output = _long_output(example)
                style = "long_reasoning"
                long_count += 1
        rows.append(
            {
                "instruction": (example.instruction or _default_instruction_for_task(example.task_type)).strip(),
                "input": example.problem,
                "output": output,
                "metadata": {
                    "example_id": example.example_id,
                    "source": example.source,
                    "style": style,
                    "difficulty": example.difficulty,
                    "task_type": example.task_type,
                    **example.metadata,
                },
            }
        )
    write_jsonl(output_path, rows)
    return {
        "rows": rows,
        "short_answer_count": short_count,
        "long_reasoning_count": long_count,
    }


def build_pack_stats(
    raw_examples: list[MathTrainingExample],
    final_examples: list[MathTrainingExample],
    exported_rows: list[dict[str, Any]],
    *,
    raw_source_distribution: dict[str, int] | None = None,
    filtered_source_distribution: dict[str, int] | None = None,
    processing_config: dict[str, Any] | None = None,
    adapter_script_path: str | Path | None = None,
    transform_script_path: str | Path | None = None,
) -> dict[str, Any]:
    source_counter = Counter(item.source for item in final_examples)
    style_counter = Counter(row.get("metadata", {}).get("style", "unknown") for row in exported_rows)
    topic_counter = Counter(item.metadata.get("topic", "unknown") for item in final_examples)
    duplicate_rate = 0.0
    if raw_examples:
        duplicate_rate = max(len(raw_examples) - len(final_examples), 0) / len(raw_examples)
    payload = {
        "sample_count": len(exported_rows),
        "source_count": len(source_counter),
        "source_distribution": dict(source_counter),
        "style_distribution": dict(style_counter),
        "topic_distribution": dict(topic_counter),
        "duplicate_rate": duplicate_rate,
    }
    if raw_source_distribution is not None:
        payload["raw_source_distribution"] = raw_source_distribution
    if filtered_source_distribution is not None:
        payload["filtered_source_distribution"] = filtered_source_distribution
    if processing_config:
        payload["processing_config"] = processing_config
    if adapter_script_path:
        payload["adapter_script_path"] = str(adapter_script_path)
    if transform_script_path:
        payload["transform_script_path"] = str(transform_script_path)
    return payload


def validate_train_config(
    train_config_path: str | Path,
    *,
    output_path: str | Path | None = None,
    effective_output_path: str | Path | None = None,
    defaults: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config_path = Path(train_config_path)
    issues: list[str] = []
    effective_config = dict(DIRECT_TRAIN_CONFIG_DEFAULTS)
    if isinstance(defaults, dict):
        for key, value in defaults.items():
            if key in TRAIN_CONFIG_ALLOWED_FIELDS and value not in (None, ""):
                effective_config[key] = value

    raw_payload: dict[str, Any] = {}
    if not config_path.exists():
        issues.append(f"train config not found: {config_path}")
    else:
        try:
            loaded = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            issues.append(f"train config is not valid JSON: {exc}")
            loaded = {}
        if not isinstance(loaded, dict):
            issues.append("train config must be a JSON object")
        else:
            raw_payload = loaded

    if raw_payload:
        unexpected = sorted(set(raw_payload) - TRAIN_CONFIG_ALLOWED_FIELDS)
        if unexpected:
            issues.append(f"train config contains unsupported keys: {unexpected}")

        int_fields = {"per_device_train_batch_size", "gradient_accumulation_steps", "cutoff_len", "max_samples"}
        for key in sorted(TRAIN_CONFIG_ALLOWED_FIELDS):
            if key not in raw_payload:
                continue
            value = raw_payload[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                issues.append(f"train config field {key} must be numeric")
                continue
            if key in int_fields and int(value) != value:
                issues.append(f"train config field {key} must be an integer")
                continue
            lower, upper = DIRECT_TRAIN_CONFIG_LIMITS[key]
            normalized = int(value) if key in int_fields else float(value)
            if normalized < lower or normalized > upper:
                issues.append(
                    f"train config field {key}={normalized} is outside allowed range [{lower}, {upper}]"
                )
                continue
            effective_config[key] = normalized

    status = "passed" if not issues else "failed"
    reason = "train config passed" if not issues else issues[0]
    payload = {
        "status": status,
        "reason": reason,
        "train_config_path": str(config_path),
        "effective_train_config_path": str(effective_output_path) if effective_output_path else "",
        "provided_keys": sorted(raw_payload.keys()),
        "effective_config": effective_config,
        "issues": issues,
    }
    if not issues and effective_output_path:
        write_json(effective_output_path, effective_config)
    if output_path:
        write_json(output_path, payload)
        payload["report_path"] = str(output_path)
    return payload


def _parse_raw_rows_seen(value: Any) -> tuple[int | None, dict[str, int] | None, str | None]:
    if value is None:
        return None, None, None
    if isinstance(value, bool):
        return None, None, "prep report raw_rows_seen must be an integer or a source->count object"
    if isinstance(value, int):
        return value, None, None
    if isinstance(value, float) and value.is_integer():
        return int(value), None, None
    if isinstance(value, dict):
        breakdown: dict[str, int] = {}
        total = 0
        for raw_key, raw_count in value.items():
            key = str(raw_key).strip()
            if not key:
                return None, None, "prep report raw_rows_seen breakdown contains an empty source key"
            if isinstance(raw_count, bool) or not isinstance(raw_count, (int, float)):
                return None, None, "prep report raw_rows_seen breakdown values must be integers"
            count = int(raw_count)
            if count != raw_count or count < 0:
                return None, None, "prep report raw_rows_seen breakdown values must be non-negative integers"
            breakdown[key] = count
            total += count
        return total, breakdown, None
    return None, None, "prep report raw_rows_seen must be an integer or a source->count object"


def validate_prepared_train_file(
    train_jsonl_path: str | Path,
    prep_report_path: str | Path | None = None,
    *,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    train_path = Path(train_jsonl_path)
    issues: list[str] = []
    row_count = 0
    selected_sources: list[str] = []
    preview_rows: list[dict[str, Any]] = []
    metadata_field_types: dict[str, tuple[str, int]] = {}

    def _record_metadata_type(field: str, value: Any, lineno: int) -> None:
        if value is None:
            return

        if field == "tags":
            if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
                issues.append(f"line {lineno} has invalid metadata.tags: expected list[str]")
                return
            current_type = "list[str]"
        elif field in {"source_id", "topic", "difficulty", "raw_id"}:
            if not isinstance(value, str):
                issues.append(f"line {lineno} has invalid metadata.{field}: expected string")
                return
            current_type = "str"
        else:
            current_type = type(value).__name__

        previous = metadata_field_types.get(field)
        if previous is None:
            metadata_field_types[field] = (current_type, lineno)
            return

        previous_type, previous_lineno = previous
        if previous_type != current_type:
            issues.append(
                f"line {lineno} metadata.{field} changed type from {previous_type} "
                f"(first seen on line {previous_lineno}) to {current_type}"
            )

    if not train_path.exists():
        issues.append(f"train file not found: {train_path}")
    else:
        for lineno, raw_line in enumerate(train_path.read_text(encoding="utf-8").splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                issues.append(f"line {lineno} is not valid JSON: {exc}")
                break
            if not isinstance(payload, dict):
                issues.append(f"line {lineno} must be a JSON object")
                break

            row_count += 1
            if len(preview_rows) < 3:
                preview_rows.append(_clip_probe_value(payload))

            for field in ("instruction", "input", "output", "metadata"):
                if field not in payload:
                    issues.append(f"line {lineno} missing field: {field}")
                    break
            else:
                instruction = payload.get("instruction")
                input_text = payload.get("input")
                output_text = payload.get("output")
                metadata = payload.get("metadata")
                if not isinstance(instruction, str) or not instruction.strip():
                    issues.append(f"line {lineno} has invalid instruction")
                if not isinstance(input_text, str):
                    issues.append(f"line {lineno} has invalid input")
                if not isinstance(output_text, str) or not output_text.strip():
                    issues.append(f"line {lineno} has invalid output")
                if not isinstance(metadata, dict):
                    issues.append(f"line {lineno} has invalid metadata")
                else:
                    source_id = metadata.get("source_id")
                    if not isinstance(source_id, str) or not source_id.strip():
                        issues.append(f"line {lineno} missing metadata.source_id")
                    else:
                        normalized_source_id = source_id.strip()
                        if normalized_source_id not in selected_sources:
                            selected_sources.append(normalized_source_id)

                    for key, value in metadata.items():
                        _record_metadata_type(str(key), value, lineno)
            if issues:
                break

    prep_report: dict[str, Any] = {}
    if prep_report_path:
        prep_path = Path(prep_report_path)
        if not prep_path.exists():
            issues.append(f"prep report not found: {prep_path}")
        else:
            try:
                prep_report = json.loads(prep_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                issues.append(f"prep report is not valid JSON: {exc}")
            if isinstance(prep_report, dict):
                for key in (
                    "selected_sources",
                    "raw_rows_seen",
                    "rows_written",
                    "duplicate_rows_removed",
                    "notes",
                ):
                    if key not in prep_report:
                        issues.append(f"prep report missing key: {key}")
                raw_rows_seen_total, _raw_rows_seen_breakdown, raw_rows_seen_issue = _parse_raw_rows_seen(
                    prep_report.get("raw_rows_seen")
                )
                if raw_rows_seen_issue:
                    issues.append(raw_rows_seen_issue)
                elif raw_rows_seen_total is None:
                    issues.append("prep report raw_rows_seen must be present")
                rows_written = prep_report.get("rows_written")
                if isinstance(rows_written, int):
                    if rows_written != row_count:
                        issues.append(
                            f"prep report rows_written={rows_written} does not match train row_count={row_count}"
                        )
                else:
                    issues.append("prep report rows_written must be an integer")
            else:
                issues.append("prep report must be a JSON object")

    if not issues and row_count <= 0:
        issues.append("train file contains zero usable rows")

    status = "passed" if not issues else "failed"
    reason = "prepared train file passed" if not issues else issues[0]
    payload = {
        "status": status,
        "reason": reason,
        "train_jsonl_path": str(train_path),
        "prep_report_path": str(prep_report_path) if prep_report_path else "",
        "row_count": row_count,
        "selected_sources": selected_sources,
        "issues": issues,
        "preview_rows": preview_rows,
    }
    if output_path:
        write_json(output_path, payload)
        payload["report_path"] = str(output_path)
    return payload


def synthesize_pack_from_prepared_train_file(
    dataset_entries: list[dict[str, Any]],
    train_jsonl_path: str | Path,
    prep_report_path: str | Path | None,
    *,
    pack_id: str,
) -> tuple[TrainPackManifest, dict[str, Any], Path]:
    train_path = Path(train_jsonl_path)
    rows = read_jsonl(train_path)
    prep_report: dict[str, Any] = {}
    if prep_report_path and Path(prep_report_path).exists():
        loaded = json.loads(Path(prep_report_path).read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            prep_report = loaded

    source_counter: Counter[str] = Counter()
    style_counter: Counter[str] = Counter()
    topic_counter: Counter[str] = Counter()
    coverage_tags: set[str] = set()
    seen_pairs: set[tuple[str, str]] = set()
    duplicate_count = 0
    entry_map = {
        str(item.get("source_id") or item.get("name") or "").strip(): item
        for item in dataset_entries
        if isinstance(item, dict)
    }

    for row in rows:
        if not isinstance(row, dict):
            continue
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        source_id = str(metadata.get("source_id") or "").strip() or "unknown"
        source_counter[source_id] += 1
        topic_counter[str(metadata.get("topic") or "unknown")] += 1

        style = str(metadata.get("style") or "").strip()
        if style not in {"short_answer", "long_reasoning"}:
            style = _derive_answer_style(str(row.get("output") or ""))
        style_counter[style] += 1

        normalized_pair = (
            normalize_text(str(row.get("instruction") or "")).lower(),
            normalize_text(str(row.get("output") or "")).lower(),
        )
        if normalized_pair in seen_pairs:
            duplicate_count += 1
        else:
            seen_pairs.add(normalized_pair)

        for tag in (entry_map.get(source_id, {}).get("coverage_tags") or []):
            coverage_tags.add(str(tag))
        raw_tags = metadata.get("tags")
        if isinstance(raw_tags, list):
            for tag in raw_tags:
                if str(tag).strip():
                    coverage_tags.add(str(tag).strip())
        elif isinstance(raw_tags, str) and raw_tags.strip():
            coverage_tags.add(raw_tags.strip())

    duplicate_rate = (duplicate_count / len(rows)) if rows else 0.0
    raw_rows_seen_total, raw_rows_seen_breakdown, _raw_rows_seen_issue = _parse_raw_rows_seen(
        prep_report.get("raw_rows_seen")
    )
    if raw_rows_seen_total is None:
        raw_rows_seen_total = len(rows)
    stats = {
        "sample_count": len(rows),
        "source_count": len(source_counter),
        "source_distribution": dict(source_counter),
        "style_distribution": dict(style_counter),
        "topic_distribution": dict(topic_counter),
        "duplicate_rate": duplicate_rate,
        "preparation_mode": "direct_jsonl",
        "prep_report_path": str(prep_report_path) if prep_report_path else "",
        "raw_rows_seen": raw_rows_seen_total,
        "rows_written": int(prep_report.get("rows_written") or len(rows)),
        "duplicate_rows_removed": int(prep_report.get("duplicate_rows_removed") or 0),
        "selected_sources": [str(item) for item in (prep_report.get("selected_sources") or [])],
        "notes": str(prep_report.get("notes") or ""),
    }
    if raw_rows_seen_breakdown:
        stats["raw_rows_seen_breakdown"] = raw_rows_seen_breakdown
    manifest = TrainPackManifest(
        pack_id=pack_id,
        source_datasets=sorted(source_counter),
        sample_count=len(rows),
        short_answer_count=int(style_counter.get("short_answer", 0)),
        long_reasoning_count=int(style_counter.get("long_reasoning", 0)),
        dedup_rule="agent_defined_in_prepare_data.py",
        answer_normalization_rule="agent_authored_direct_jsonl",
        format="alpaca",
        output_path=str(train_path),
        source_weights={},
        coverage_tags=sorted(coverage_tags),
        strategy={
            "preparation_mode": "direct_jsonl",
            "prep_report_path": str(prep_report_path) if prep_report_path else "",
        },
    )
    return manifest, stats, train_path


def prepare_dataset_probe(
    dataset_entries: list[dict[str, Any]],
    output_dir: str | Path,
    *,
    max_rows_per_source: int = DEFAULT_PROBE_MAX_ROWS,
    materialize_max_rows: int | None = DEFAULT_PROBE_MAX_ROWS,
    data_access_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    probe_dir = Path(output_dir)
    probe_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = probe_dir / "_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    sources: list[dict[str, Any]] = []
    total_sample_rows = 0
    for entry in dataset_entries:
        source_id = str(entry.get("source_id") or entry.get("name") or "unknown_source")
        task_type = _infer_task_type(entry, entry.get("task_type"))
        full_local_path = _get_entry_full_local_path(entry)
        if full_local_path and not _materialized_path_has_rows(Path(full_local_path)):
            full_local_path = ""
        probe_sample_rows_path = _get_entry_probe_sample_rows_path(entry)
        if probe_sample_rows_path and not _materialized_path_has_rows(Path(probe_sample_rows_path)):
            probe_sample_rows_path = ""

        probe_source_path = full_local_path
        if not probe_source_path and not probe_sample_rows_path:
            LOGGER.warning(
                "Probe requested for %s without full_local_path; materializing a temporary probe cache with max_rows=%s",
                source_id,
                "all" if materialize_max_rows is None else materialize_max_rows,
            )
            probe_source_path = materialize_dataset_entry(
                dict(entry),
                cache_dir,
                max_rows=materialize_max_rows,
                data_access_config=data_access_config,
            )
            if materialize_max_rows is None and probe_source_path and _materialized_path_has_rows(Path(probe_source_path)):
                full_local_path = probe_source_path

        if full_local_path and _materialized_path_has_rows(Path(full_local_path)):
            _set_entry_full_local_path(entry, full_local_path)
        else:
            full_local_path = ""
            _set_entry_full_local_path(entry, "")

        rows: list[dict[str, Any]] = []
        if probe_sample_rows_path:
            rows = _iter_rows_limited(Path(probe_sample_rows_path), max_rows_per_source)
        elif probe_source_path:
            rows = _iter_rows_limited(Path(probe_source_path), max_rows_per_source)
        total_sample_rows += len(rows)

        sample_rows_path = probe_dir / f"{_safe_source_filename(source_id)}_sample_rows.jsonl"
        write_jsonl(sample_rows_path, rows)
        if rows:
            probe_sample_rows_path = str(sample_rows_path)
        else:
            probe_sample_rows_path = ""
        _set_entry_probe_sample_rows_path(entry, probe_sample_rows_path)
        summary = _summarize_probe_rows(rows)
        sources.append(
            {
                "source_id": source_id,
                "task_type": task_type,
                "full_local_path": full_local_path,
                "local_path": full_local_path,
                "probe_sample_rows_path": probe_sample_rows_path,
                "sample_rows_path": str(sample_rows_path),
                "coverage_tags": [str(tag) for tag in (entry.get("coverage_tags") or [])],
                **summary,
            }
        )

    payload = {
        "probe_dir": str(probe_dir),
        "max_rows_per_source": max_rows_per_source,
        "source_count": len(sources),
        "total_sample_rows": total_sample_rows,
        "sources": sources,
    }
    write_json(probe_dir / "probe_summary.json", payload)
    return payload


def run_adapter_dry_run(
    dataset_entries: list[dict[str, Any]],
    probe_payload: dict[str, Any],
    *,
    processing_config: dict[str, Any] | None = None,
    source_weights: dict[str, float] | None = None,
    adapter_script_path: str | Path | None = None,
    transform_script_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    processing_config = processing_config or {}
    source_weights = source_weights or {}
    selected_entries = _iter_selected_dataset_entries(dataset_entries, processing_config, source_weights)
    source_probe_map = {
        str(item.get("source_id") or ""): item
        for item in (probe_payload.get("sources") or [])
        if isinstance(item, dict)
    }

    source_reports: list[dict[str, Any]] = []
    aggregate_drop_reasons: Counter[str] = Counter()
    total_raw_rows = 0
    total_adapted_examples = 0
    total_filtered_examples = 0

    for _, source_id, task_type, _ in selected_entries:
        probe_info = source_probe_map.get(source_id, {})
        sample_rows_path = str(probe_info.get("sample_rows_path") or "").strip()
        rows = _iter_rows(Path(sample_rows_path)) if sample_rows_path and Path(sample_rows_path).exists() else []
        total_raw_rows += len(rows)

        if adapter_script_path:
            examples, adapter_report = _run_custom_data_adapter_on_rows(
                rows,
                source_id=source_id,
                task_type=task_type,
                raw_path=sample_rows_path,
                adapter_script_path=adapter_script_path,
            )
            parse_mode = "custom_adapter"
        else:
            examples = load_task_aligned_examples(sample_rows_path, source_id=source_id, task_type=task_type) if sample_rows_path else []
            adapter_report = {
                "raw_rows": len(rows),
                "inspected_rows": len(rows),
                "after_raw_transform": len(rows),
                "after_to_example": len(examples),
                "after_validate": len(examples),
                "usable_examples": len(examples),
                "drop_reasons": (
                    {"builtin_parser:returned_zero_examples": len(rows)} if rows and not examples else {}
                ),
                "example_previews": [_clip_probe_value(item.to_dict()) for item in examples[:3]],
                "failure_previews": [],
            }
            parse_mode = "builtin_parser"

        total_adapted_examples += len(examples)
        examples = apply_transform_script(examples, transform_script_path)
        transformed_count = len(examples)
        examples = _apply_processing_config(examples, processing_config)
        filtered_count = len(examples)
        total_filtered_examples += filtered_count

        drop_reasons = dict(adapter_report.get("drop_reasons") or {})
        aggregate_drop_reasons.update(drop_reasons)
        source_reports.append(
            {
                "source_id": source_id,
                "task_type": task_type,
                "parse_mode": parse_mode,
                "sample_rows_path": sample_rows_path,
                "sample_row_count": len(rows),
                "adapted_example_count": int(adapter_report.get("usable_examples") or len(examples)),
                "transformed_example_count": transformed_count,
                "filtered_example_count": filtered_count,
                "drop_reasons": drop_reasons,
                "example_previews": adapter_report.get("example_previews") or [],
                "failure_previews": adapter_report.get("failure_previews") or [],
                "schema_keys": list(probe_info.get("schema_keys") or []),
            }
        )

    selected_source_count = len(selected_entries)
    min_examples = max(1, selected_source_count)
    filtered_rate = (total_filtered_examples / total_raw_rows) if total_raw_rows else 0.0

    if selected_source_count <= 0:
        status = "failed"
        reason = "strategy selected zero sources"
    elif total_raw_rows <= 0:
        status = "failed"
        reason = "probe materialized zero rows"
    elif total_filtered_examples <= 0:
        status = "failed"
        reason = "adapter produced zero usable training examples on probe rows"
    elif total_filtered_examples < min_examples or filtered_rate < DEFAULT_ADAPTER_REPAIR_MIN_FILTERED_RATE:
        status = "needs_repair"
        reason = "adapter usable rate is too low on probe rows"
    else:
        status = "passed"
        reason = "adapter probe passed"

    payload = {
        "status": status,
        "reason": reason,
        "passed": status == "passed",
        "selected_source_count": selected_source_count,
        "raw_row_count": total_raw_rows,
        "adapted_example_count": total_adapted_examples,
        "filtered_example_count": total_filtered_examples,
        "filtered_example_rate": filtered_rate,
        "adapter_script_path": str(adapter_script_path) if adapter_script_path else "",
        "transform_script_path": str(transform_script_path) if transform_script_path else "",
        "drop_reasons": dict(aggregate_drop_reasons),
        "source_reports": source_reports,
    }
    if output_path:
        write_json(output_path, payload)
    return payload


def validate_adapter(
    dataset_entries: list[dict[str, Any]],
    probe_payload: dict[str, Any],
    *,
    processing_config: dict[str, Any] | None = None,
    source_weights: dict[str, float] | None = None,
    adapter_script_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    detailed = run_adapter_dry_run(
        dataset_entries,
        probe_payload,
        processing_config=processing_config,
        source_weights=source_weights,
        adapter_script_path=adapter_script_path,
        transform_script_path=None,
        output_path=output_path,
    )
    return {
        "status": str(detailed.get("status") or "failed"),
        "raw_rows": int(detailed.get("raw_row_count") or 0),
        "usable_rows": int(detailed.get("filtered_example_count") or 0),
        "reason": str(detailed.get("reason") or "adapter validation failed"),
        "report_path": str(output_path) if output_path else "",
    }


def build_train_pack(
    dataset_entries: list[dict[str, Any]],
    output_dir: str | Path,
    *,
    pack_id: str,
    max_samples: int = 512,
    short_answer_ratio: float = 0.5,
    source_weights: dict[str, float] | None = None,
    processing_config: dict[str, Any] | None = None,
    adapter_script_path: str | Path | None = None,
    transform_script_path: str | Path | None = None,
    data_access_config: dict[str, Any] | None = None,
) -> tuple[TrainPackManifest, dict[str, Any], Path]:
    all_examples: list[MathTrainingExample] = []
    coverage_tags: set[str] = set()
    source_weights = source_weights or {}
    processing_config = processing_config or {}
    dedup_keep_mode = str(processing_config.get("dedup_keep_mode") or "short_and_long").strip() or "short_and_long"
    pack_dir = Path(output_dir)
    raw_source_distribution: Counter[str] = Counter()
    filtered_source_distribution: Counter[str] = Counter()
    for entry, source_id, task_type, raw_weight in _iter_selected_dataset_entries(
        dataset_entries,
        processing_config,
        source_weights,
    ):
        local_path = _get_entry_full_local_path(entry)
        if not local_path:
            materialize_max_rows = _estimate_materialization_max_rows(
                max_samples=max_samples,
                source_weight=raw_weight,
                processing_config=processing_config,
            )
            LOGGER.info(
                "full_local_path missing for %s; falling back to in-build materialization with max_rows=%d (weight=%.3f, max_samples=%d)",
                source_id,
                materialize_max_rows,
                raw_weight,
                max_samples,
            )
            local_path = materialize_dataset_entry(
                entry,
                pack_dir / "_cache",
                max_rows=materialize_max_rows,
                data_access_config=data_access_config,
            )
            if local_path:
                _set_entry_full_local_path(entry, local_path)
        if not local_path:
            continue
        if adapter_script_path:
            examples, _ = _run_custom_data_adapter_on_rows(
                _iter_rows(Path(local_path)),
                source_id=source_id,
                task_type=task_type,
                raw_path=local_path,
                adapter_script_path=adapter_script_path,
            )
        else:
            examples = load_task_aligned_examples(local_path, source_id=source_id, task_type=task_type)
        raw_source_distribution[source_id] += len(examples)
        examples = apply_transform_script(examples, transform_script_path)
        examples = _apply_processing_config(examples, processing_config)
        filtered_source_distribution[source_id] += len(examples)
        multiplier = max(int(raw_weight * 10), 1)
        weighted_examples = (examples * multiplier)[:max_samples]
        all_examples.extend(weighted_examples)
        for tag in entry.get("coverage_tags", []):
            coverage_tags.add(str(tag))

    deduped = deduplicate_examples(all_examples, keep_mode=dedup_keep_mode)[:max_samples]
    pack_dir.mkdir(parents=True, exist_ok=True)
    alpaca_path = pack_dir / "alpaca_train.jsonl"
    exported = export_alpaca_dataset(
        deduped,
        alpaca_path,
        short_answer_ratio=short_answer_ratio,
    )
    stats = build_pack_stats(
        all_examples,
        deduped,
        exported["rows"],
        raw_source_distribution=dict(raw_source_distribution),
        filtered_source_distribution=dict(filtered_source_distribution),
        processing_config=processing_config,
        adapter_script_path=adapter_script_path,
        transform_script_path=transform_script_path,
    )
    has_tool_using = any(_is_tool_using_task(item.task_type) for item in deduped)
    dedup_rule = "exact(normalized_problem, normalized_final_answer), keep up to one short + one long"
    answer_normalization_rule = "strip boxed/math markdown, collapse whitespace, trim punctuation"
    if has_tool_using:
        dedup_rule = "exact(normalized_input, normalized_output), keep first occurrence only"
        answer_normalization_rule = "collapse whitespace and strip special end-of-text tokens"
    elif dedup_keep_mode == "short_only":
        dedup_rule = "exact(normalized_problem, normalized_final_answer), keep short only"
    elif dedup_keep_mode == "long_only":
        dedup_rule = "exact(normalized_problem, normalized_final_answer), keep long only"
    elif dedup_keep_mode == "first":
        dedup_rule = "exact(normalized_problem, normalized_final_answer), keep first occurrence only"
    manifest = TrainPackManifest(
        pack_id=pack_id,
        source_datasets=sorted({item.source for item in deduped}),
        sample_count=len(exported["rows"]),
        short_answer_count=exported["short_answer_count"],
        long_reasoning_count=exported["long_reasoning_count"],
        dedup_rule=dedup_rule,
        answer_normalization_rule=answer_normalization_rule,
        format="alpaca",
        output_path=str(alpaca_path),
        source_weights=source_weights,
        coverage_tags=sorted(coverage_tags),
        strategy={
            "max_samples": max_samples,
            "short_answer_ratio": short_answer_ratio,
            "processing_config": processing_config,
            "adapter_script_path": str(adapter_script_path) if adapter_script_path else "",
            "transform_script_path": str(transform_script_path) if transform_script_path else "",
        },
    )
    return manifest, stats, alpaca_path
