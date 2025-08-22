from __future__ import annotations

from decimal import Decimal, getcontext, ROUND_DOWN
from typing import Tuple

from nge_trader.config.settings import Settings


def _q(value: float, step: float) -> float:
    try:
        if not step or step <= 0:
            return float(value)
        getcontext().prec = 28
        v = Decimal(str(value))
        st = Decimal(str(step))
        q = (v / st).to_integral_value(rounding=ROUND_DOWN) * st
        return float(q)
    except Exception:
        return float(value)


def round_price_qty(symbol: str, price: float | None, qty: float) -> Tuple[float | None, float]:
    s = Settings()
    # lot size per symbol
    try:
        import json as _json
        lot_map = _json.loads(str(getattr(s, "lot_size_per_symbol", "{}"))) if getattr(s, "lot_size_per_symbol", None) else {}
    except Exception:
        lot_map = {}
    qty_step = float(lot_map.get(symbol.upper()) or (s.qty_step or 0.0) or 0.0)
    q_qty = _q(float(qty), qty_step) if qty_step else float(qty)
    # tick
    tick = float(getattr(s, "price_tick", 0.0) or 0.0)
    q_px = _q(float(price), tick) if (price is not None and tick) else (float(price) if price is not None else None)
    return q_px, q_qty


