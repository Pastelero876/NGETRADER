from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TradeIntent:
    symbol: str
    side: str  # "buy"|"sell"|"no-trade"
    confidence: float
    horizon_min: int
    edge_bps: float
    reason: str


