"""HF Sandbox Server – FastAPI service with rate limiting and concurrency control."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from contextlib import asynccontextmanager
from functools import partial
from typing import Any, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from . import config as cfg
from .rate_limiter import TokenBucketRateLimiter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
LOGGER = logging.getLogger("hf_sandbox_server")

# ── Global state ─────────────────────────────────────────────────────

rate_limiter: TokenBucketRateLimiter | None = None
semaphore: asyncio.Semaphore | None = None

# metrics
_metrics: dict[str, int] = {
    "total_requests": 0,
    "search_cache_hits": 0,
    "errors": 0,
}
_start_time: float = 0.0

# search cache: key -> (timestamp, result)
_search_cache: dict[str, tuple[float, str]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rate_limiter, semaphore, _start_time
    rate_limiter = TokenBucketRateLimiter(rpm=cfg.RATE_LIMIT_RPM)
    semaphore = asyncio.Semaphore(cfg.MAX_CONCURRENT)
    _start_time = time.time()
    LOGGER.info(
        "HF Sandbox started – RPM=%d, max_concurrent=%d, cache_ttl=%ds, port=%d",
        cfg.RATE_LIMIT_RPM,
        cfg.MAX_CONCURRENT,
        cfg.SEARCH_CACHE_TTL,
        cfg.PORT,
    )
    yield


app = FastAPI(title="HF Sandbox Server", lifespan=lifespan)


# ── Request / Response models ────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str
    limit: int = 100
    author: Optional[str] = None

class InspectRequest(BaseModel):
    dataset_id: str
    config: Optional[str] = None

class ConfigsRequest(BaseModel):
    dataset_id: str

class SplitsRequest(BaseModel):
    dataset_id: str
    config: Optional[str] = None

class ReadmeRequest(BaseModel):
    dataset_id: str

class SampleRequest(BaseModel):
    dataset_id: str
    config: Optional[str] = None
    split: Optional[str] = None
    num_samples: int = 5

class DownloadRequest(BaseModel):
    dataset_id: str
    output_dir: str = "./downloaded_datasets"

class MaterializeRequest(BaseModel):
    dataset: str
    config: Optional[str] = None
    split: Optional[str] = None
    max_rows: int = 2048
    output_dir: str = ""


# ── Helpers ──────────────────────────────────────────────────────────

def _cache_key(prefix: str, **kwargs: Any) -> str:
    raw = f"{prefix}:" + ":".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
    return hashlib.md5(raw.encode()).hexdigest()


def _get_cached(key: str) -> str | None:
    if key in _search_cache:
        ts, val = _search_cache[key]
        if time.time() - ts < cfg.SEARCH_CACHE_TTL:
            return val
        del _search_cache[key]
    return None


def _set_cached(key: str, val: str) -> None:
    _search_cache[key] = (time.time(), val)
    if len(_search_cache) > 500:
        oldest = min(_search_cache, key=lambda k: _search_cache[k][0])
        del _search_cache[oldest]


async def _run_with_limits(fn, *args, **kwargs) -> Any:
    _metrics["total_requests"] += 1
    await rate_limiter.acquire()
    async with semaphore:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, partial(fn, *args, **kwargs))


# ── Endpoints ────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    uptime = time.time() - _start_time if _start_time else 0
    return {
        "status": "ok",
        "uptime_seconds": round(uptime, 1),
        "config": {
            "rate_limit_rpm": cfg.RATE_LIMIT_RPM,
            "max_concurrent": cfg.MAX_CONCURRENT,
            "search_cache_ttl": cfg.SEARCH_CACHE_TTL,
            "hf_endpoint": cfg.HF_ENDPOINT,
        },
        "metrics": {**_metrics, "cache_size": len(_search_cache)},
    }


@app.post("/search")
async def search(req: SearchRequest):
    key = _cache_key("search", query=req.query, limit=req.limit, author=req.author or "")
    cached = _get_cached(key)
    if cached is not None:
        _metrics["search_cache_hits"] += 1
        return {"text": cached, "cached": True}

    from .hf_ops import search_datasets
    text = await _run_with_limits(search_datasets, req.query, req.limit, req.author)
    _set_cached(key, text)
    return {"text": text, "cached": False}


@app.post("/inspect")
async def inspect(req: InspectRequest):
    from .hf_ops import inspect_dataset
    text = await _run_with_limits(inspect_dataset, req.dataset_id, req.config)
    return {"text": text}


@app.post("/configs")
async def configs(req: ConfigsRequest):
    from .hf_ops import get_dataset_configs
    text = await _run_with_limits(get_dataset_configs, req.dataset_id)
    return {"text": text}


@app.post("/splits")
async def splits(req: SplitsRequest):
    from .hf_ops import get_dataset_splits
    text = await _run_with_limits(get_dataset_splits, req.dataset_id, req.config)
    return {"text": text}


@app.post("/readme")
async def readme(req: ReadmeRequest):
    from .hf_ops import get_dataset_readme
    text = await _run_with_limits(get_dataset_readme, req.dataset_id)
    return {"text": text}


@app.post("/sample")
async def sample(req: SampleRequest):
    from .hf_ops import get_dataset_sample
    text = await _run_with_limits(
        get_dataset_sample, req.dataset_id, req.config, req.split, req.num_samples
    )
    return {"text": text}


@app.post("/download")
async def download(req: DownloadRequest):
    from .hf_ops import download_dataset
    text = await _run_with_limits(download_dataset, req.dataset_id, req.output_dir)
    return {"text": text}


@app.post("/materialize")
async def materialize(req: MaterializeRequest):
    from .hf_ops import materialize_dataset
    result = await _run_with_limits(
        materialize_dataset, req.dataset, req.config, req.split, req.max_rows, req.output_dir
    )
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "hf_sandbox_server.server:app",
        host=cfg.HOST,
        port=cfg.PORT,
        workers=1,
        timeout_keep_alive=620,
    )
