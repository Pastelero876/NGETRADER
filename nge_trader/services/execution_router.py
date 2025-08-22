from __future__ import annotations

from typing import Any, Dict

from nge_trader.services.exec_algos import (
    estimate_fill_probability,
)
from nge_trader.adapters.lowlat_provider import LowLatencyProvider
from nge_trader.services.oms import (
    place_limit_order_post_only,
    place_limit_order_marketable,
    place_market_order,
)


def route_order(broker: Any, symbol: str, side: str, quantity: float, prob_threshold: float = 0.7, band_bps: float = 5.0) -> Dict[str, Any]:
    """Router maker-first con fallback a limit marketable/market.

    - Si prob_fill ≥ threshold → limit marketable (no post-only)
    - Si prob_fill < threshold → post-only peg a microprice +/- band_bps
    """
    llp = LowLatencyProvider()
    mp = llp.get_microprice(symbol) or 0.0
    if mp <= 0:
        return place_market_order(broker, symbol, side, float(quantity))
    bps = abs(float(band_bps)) / 10000.0
    if side.lower() == "buy":
        limit_px = float(mp) * (1.0 - bps)
    else:
        limit_px = float(mp) * (1.0 + bps)
    p = estimate_fill_probability(symbol, side, float(limit_px))
    if float(p) >= float(prob_threshold):
        return place_limit_order_marketable(broker, symbol, side, float(quantity), float(limit_px), "GTC")
    return place_limit_order_post_only(broker, symbol, side, float(quantity), float(limit_px), "GTC")


