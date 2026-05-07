from __future__ import annotations

import asyncio
import time


class TokenBucketRateLimiter:
    """Token-bucket rate limiter that queues callers instead of rejecting."""

    def __init__(self, rpm: int = 30):
        self.interval = 60.0 / max(rpm, 1)
        self.tokens = float(rpm)
        self.max_tokens = float(rpm)
        self.last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            self._refill()
            while self.tokens < 1:
                wait = self.interval - (time.monotonic() - self.last_refill) % self.interval
                await asyncio.sleep(max(wait, 0.05))
                self._refill()
            self.tokens -= 1

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.max_tokens, self.tokens + elapsed / self.interval)
        self.last_refill = now
