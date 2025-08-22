from __future__ import annotations

"""Data feed con failover A/B.

Usa feed A (principal) y conmute a B (fallback) si data_stale>1s o dislocation>thr.
"""

import time
from typing import Any, Dict


class DataFeed:
    def __init__(self, feed_a: Any, feed_b: Any, dislocation_bps_threshold: float = 10.0, staleness_ms_limit: int = 1000) -> None:
        self.feed_a = feed_a
        self.feed_b = feed_b
        self.dislocation_bps_threshold = float(dislocation_bps_threshold)
        self.staleness_ms_limit = int(staleness_ms_limit)
        self._using = "A"

    def _is_stale(self, q: Dict[str, Any]) -> bool:
        try:
            ts = float(q.get("ts") or 0.0)
            return (time.time() - ts) * 1000.0 > self.staleness_ms_limit
        except Exception:
            return True

    def _dislocation_bps(self, qa: Dict[str, Any], qb: Dict[str, Any]) -> float:
        try:
            a_mid = (float(qa.get("bid") or 0.0) + float(qa.get("ask") or 0.0)) * 0.5
            b_mid = (float(qb.get("bid") or 0.0) + float(qb.get("ask") or 0.0)) * 0.5
            if a_mid <= 0 or b_mid <= 0:
                return 0.0
            return abs(a_mid - b_mid) / ((a_mid + b_mid) * 0.5) * 10000.0
        except Exception:
            return 0.0

    def get_quote(self, symbol: str) -> Dict[str, Any]:
        qa = self.feed_a.get_quote(symbol)
        qb = None
        try:
            qb = self.feed_b.get_quote(symbol)
        except Exception:
            qb = None
        failover = False
        if qa is None or self._is_stale(qa):
            failover = True
        elif qb is not None and self._dislocation_bps(qa, qb) > self.dislocation_bps_threshold:
            failover = True
        if failover and qb is not None:
            self._using = "B"
            return qb
        self._using = "A"
        return qa

    @property
    def using(self) -> str:
        return self._using


