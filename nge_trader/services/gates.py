from __future__ import annotations


def is_trade_viable(edge_bps: float, fees_bps: float, half_spread_bps: float, buffer_bps: float = 2.0) -> bool:
    return float(edge_bps) >= (float(fees_bps) + float(half_spread_bps) + float(buffer_bps))


def time_skew_ok(skew_ms: int, max_ms: int = 100) -> bool:
    return int(skew_ms) <= int(max_ms)


def market_open_ok(is_open: bool) -> bool:
    return bool(is_open)


