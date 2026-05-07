"""HF Sandbox Server configuration – all values from env vars."""

from __future__ import annotations

import os


# ── HuggingFace ──────────────────────────────────────────────────────
HF_ENDPOINT: str = os.getenv("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")
HF_TOKEN: str = os.getenv("HF_TOKEN", "")
HF_HOME: str = os.getenv("HF_HOME", "${HF_HOME}")
HF_DATASETS_CACHE: str = os.getenv(
    "HF_DATASETS_CACHE", os.path.join(HF_HOME, "datasets")
)

# ── Shared NFS cache (materialized JSONL files) ─────────────────────
SHARED_CACHE_DIR: str = os.getenv(
    "SHARED_CACHE_DIR",
    os.getenv(
        "MATH_PT_SHARED_CACHE", "/data/yaxindu/datascientist/math_posttrain_cache"
    ),
)

# ── Rate limiting ────────────────────────────────────────────────────
RATE_LIMIT_RPM: int = int(os.getenv("HF_RATE_LIMIT_RPM", "30"))
MAX_CONCURRENT: int = int(os.getenv("HF_MAX_CONCURRENT", "3"))

# ── Search cache ─────────────────────────────────────────────────────
SEARCH_CACHE_TTL: int = int(os.getenv("HF_SEARCH_CACHE_TTL", "600"))  # seconds

# ── Server ───────────────────────────────────────────────────────────
HOST: str = os.getenv("SANDBOX_HOST", "0.0.0.0")
PORT: int = int(os.getenv("SANDBOX_PORT", "8899"))
