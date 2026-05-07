"""Message deduplication

In-memory dict implementation with TTL expiration and capacity limits.
"""

from __future__ import annotations

import threading
import time
import logging

logger = logging.getLogger(__name__)

# Default parameters
_DEFAULT_TTL = 30 * 60  # 30 minutes
_DEFAULT_MAX_SIZE = 1000
_DEFAULT_CLEANUP_INTERVAL = 5 * 60  # 5 minutes


class MessageDedup:
    """Message deduplication handler."""

    def __init__(
        self,
        ttl: float = _DEFAULT_TTL,
        max_size: int = _DEFAULT_MAX_SIZE,
        cleanup_interval: float = _DEFAULT_CLEANUP_INTERVAL,
    ):
        self._store: dict[str, float] = {}
        self._lock = threading.Lock()
        self._ttl = ttl
        self._max_size = max_size
        self._cleanup_interval = cleanup_interval
        self._last_cleanup = time.monotonic()

    def try_record_message(self, message_id: str, scope: str = "default") -> bool:
        """Try to record a message.

        Args:
            message_id: Feishu message ID.
            scope: Scope (e.g. chat_id), used to isolate deduplication across different contexts.

        Returns:
            True if this is a new message (recorded), False if it is a duplicate.
        """
        key = f"{scope}:{message_id}"
        now = time.monotonic()

        with self._lock:
            # Periodic cleanup
            if now - self._last_cleanup > self._cleanup_interval:
                self._cleanup(now)

            if key in self._store:
                logger.debug("Duplicate message: %s", key)
                return False

            self._store[key] = now

            # Force cleanup on capacity overflow
            if len(self._store) > self._max_size:
                self._cleanup(now)

            return True

    def _cleanup(self, now: float) -> None:
        """Clean up expired entries (caller must hold _lock)."""
        expired = [k for k, ts in self._store.items() if now - ts > self._ttl]
        for k in expired:
            del self._store[k]

        # If still over capacity, delete the oldest entries sorted by time
        if len(self._store) > self._max_size:
            sorted_keys = sorted(self._store, key=self._store.get)  # type: ignore[arg-type]
            excess = len(self._store) - self._max_size
            for k in sorted_keys[:excess]:
                del self._store[k]

        self._last_cleanup = now
        logger.debug("Dedup cleanup: %d entries remaining", len(self._store))
