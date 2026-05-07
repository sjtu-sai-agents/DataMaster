"""
Improved data.py with better SSL error handling and connection management.

Key improvements:
1. Session with connection pooling and retry adapter
2. Rate limiting between requests
3. Better backoff strategy
4. Connection timeout settings
"""

from __future__ import annotations

import time
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Existing constants
DATASETS_SERVER_BASE = "https://datasets-server.huggingface.co"
DATASETS_SERVER_RETRIES = 4
DATASETS_SERVER_BACKOFF_SECONDS = 1.0
MIN_REQUEST_INTERVAL_SECONDS = 0.2  # NEW: Minimum time between requests


# NEW: Create a session with connection pooling and retry logic
def _create_datasets_server_session() -> requests.Session:
    """
    Create a requests session with:
    - Connection pooling (reuse connections)
    - Automatic retry with exponential backoff
    - Connection timeout settings
    """
    session = requests.Session()

    # Configure retry strategy
    retry_strategy = Retry(
        total=DATASETS_SERVER_RETRIES,
        backoff_factor=DATASETS_SERVER_BACKOFF_SECONDS,
        status_forcelist=[429, 500, 502, 503, 504],  # Retry on these HTTP status codes
        allowed_methods=["GET"],  # Only retry GET requests
        raise_on_status=False,  # Don't raise exception, let us handle it
    )

    # Mount adapter with retry strategy
    adapter = HTTPAdapter(
        max_retries=retry_strategy,
        pool_connections=10,  # Number of connection pools
        pool_maxsize=20,  # Max connections per pool
    )

    session.mount("https://", adapter)
    session.mount("http://", adapter)

    # Set default headers
    session.headers.update({
        'Connection': 'keep-alive',
        'User-Agent': 'math-posttrain-datatree/1.0',
    })

    return session


# Global session instance (reuse connections across requests)
_DATASETS_SERVER_SESSION: requests.Session | None = None
_LAST_REQUEST_TIME: float = 0.0


def _get_datasets_server_session() -> requests.Session:
    """Get or create the global datasets-server session."""
    global _DATASETS_SERVER_SESSION
    if _DATASETS_SERVER_SESSION is None:
        _DATASETS_SERVER_SESSION = _create_datasets_server_session()
    return _DATASETS_SERVER_SESSION


def _rate_limit_request() -> None:
    """
    Enforce minimum time between requests to avoid overwhelming the server.
    This helps prevent SSL EOF errors caused by too many concurrent connections.
    """
    global _LAST_REQUEST_TIME
    current_time = time.time()
    time_since_last = current_time - _LAST_REQUEST_TIME

    if time_since_last < MIN_REQUEST_INTERVAL_SECONDS:
        sleep_time = MIN_REQUEST_INTERVAL_SECONDS - time_since_last
        time.sleep(sleep_time)

    _LAST_REQUEST_TIME = time.time()


def _datasets_server_get(
    endpoint: str,
    *,
    params: dict[str, Any],
    timeout: int | tuple[int, int],
    dataset_id: str,
) -> requests.Response:
    """
    IMPROVED VERSION: Make a GET request to datasets-server with:
    - Connection pooling via session
    - Rate limiting between requests
    - Better timeout handling (connect timeout + read timeout)
    - Exponential backoff on failures

    Args:
        endpoint: API endpoint (e.g., "rows", "splits")
        params: Query parameters
        timeout: Request timeout in seconds, or tuple of (connect_timeout, read_timeout)
        dataset_id: Dataset identifier for logging

    Returns:
        requests.Response object

    Raises:
        Exception: If all retries are exhausted
    """
    import logging
    LOGGER = logging.getLogger(__name__)

    # Use session for connection pooling
    session = _get_datasets_server_session()

    # Convert timeout to tuple if single value provided
    if isinstance(timeout, int):
        timeout = (10, timeout)  # (connect_timeout, read_timeout)

    last_error: Exception | None = None

    for attempt in range(1, DATASETS_SERVER_RETRIES + 1):
        try:
            # Rate limit to avoid overwhelming the server
            _rate_limit_request()

            # Make the request
            resp = session.get(
                f"{DATASETS_SERVER_BASE}/{endpoint}",
                params=params,
                timeout=timeout,
            )
            resp.raise_for_status()
            return resp

        except requests.exceptions.SSLError as exc:
            last_error = exc
            if attempt >= DATASETS_SERVER_RETRIES:
                break

            # Longer backoff for SSL errors
            backoff = DATASETS_SERVER_BACKOFF_SECONDS * attempt * 2  # Double the backoff for SSL errors
            LOGGER.info(
                "datasets-server %s SSL error for %s (attempt %d/%d): %s; "
                "retrying after %.1fs",
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
                "datasets-server %s request failed for %s (attempt %d/%d): %s; "
                "retrying after %.1fs",
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


# Example usage comparison:
"""
BEFORE (in data.py line 126-131):
    resp = requests.get(
        f"{DATASETS_SERVER_BASE}/{endpoint}",
        params=params,
        timeout=timeout,
    )

AFTER (improved version):
    resp = session.get(  # Reuse connection
        f"{DATASETS_SERVER_BASE}/{endpoint}",
        params=params,
        timeout=(10, timeout),  # Separate connect/read timeout
    )
    # Plus: rate limiting, better retry logic, connection pooling
"""


if __name__ == "__main__":
    # Test the improved implementation
    print("Testing improved SSL error handling...")

    # Test with actual dataset
    try:
        resp = _datasets_server_get(
            "splits",
            params={"dataset": "openai/gsm8k"},
            timeout=30,
            dataset_id="openai/gsm8k",
        )
        print(f"✓ Success! Status: {resp.status_code}")
        print(f"  Response: {resp.json()}")
    except Exception as e:
        print(f"✗ Failed: {e}")
