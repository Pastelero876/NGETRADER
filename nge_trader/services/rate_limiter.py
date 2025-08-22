from __future__ import annotations

import time
from dataclasses import dataclass
from threading import Lock
from typing import Dict


@dataclass
class TokenBucket:
    capacity: float
    refill_rate_per_sec: float
    tokens: float
    last_refill_ts: float

    def try_consume(self, amount: float = 1.0) -> bool:
        now = time.time()
        elapsed = max(now - self.last_refill_ts, 0.0)
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate_per_sec)
        self.last_refill_ts = now
        if self.tokens >= amount:
            self.tokens -= amount
            return True
        return False


class GlobalRateLimiter:
    """Rate limiter global en memoria, basado en token-bucket por clave.

    Uso:
        RateLimiter.get().configure("orders", capacity=60, refill_per_sec=1)
        RateLimiter.get().acquire("orders")
    """

    _instance: "GlobalRateLimiter | None" = None
    _lock = Lock()

    def __init__(self) -> None:
        self._buckets: Dict[str, TokenBucket] = {}
        self._mtx = Lock()

    @classmethod
    def get(cls) -> "GlobalRateLimiter":
        with cls._lock:
            if cls._instance is None:
                cls._instance = GlobalRateLimiter()
            return cls._instance

    def configure(self, key: str, capacity: int, refill_per_sec: float) -> None:
        with self._mtx:
            now = time.time()
            self._buckets[key] = TokenBucket(
                capacity=float(capacity),
                refill_rate_per_sec=float(refill_per_sec),
                tokens=float(capacity),
                last_refill_ts=now,
            )

    def acquire(self, key: str, amount: float = 1.0) -> bool:
        with self._mtx:
            bucket = self._buckets.get(key)
            if bucket is None:
                # por defecto: 60 op/min
                self.configure(key, capacity=60, refill_per_sec=1.0)
                bucket = self._buckets[key]
            return bucket.try_consume(amount)

    def remaining(self, key: str) -> float:
        with self._mtx:
            bucket = self._buckets.get(key)
            if bucket is None:
                return 0.0
            # actualizar para cálculo
            bucket.try_consume(0.0)
            return float(bucket.tokens)

    def capacity(self, key: str) -> float:
        with self._mtx:
            bucket = self._buckets.get(key)
            return float(bucket.capacity) if bucket is not None else 0.0

