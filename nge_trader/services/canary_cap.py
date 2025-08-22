from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

try:
    from prometheus_client import Gauge  # type: ignore
except Exception:  # pragma: no cover
    Gauge = None  # type: ignore

canary_notional_used_pct = Gauge("canary_notional_used_pct", "% notional canario usado (día)") if Gauge else None  # type: ignore


def _today_utc():
    return datetime.now(timezone.utc).date()


def compute_used_pct(db) -> float:
    # TODO: reemplazar por consultas reales a fills/orders del día (canary=true)
    # Debe devolver (notional_canary / notional_total_del_dia)
    return 0.0


def gate(db, max_pct: float) -> bool:
    used = float(compute_used_pct(db))
    try:
        if canary_notional_used_pct is not None:  # type: ignore
            canary_notional_used_pct.set(used)  # type: ignore[attr-defined]
    except Exception:
        pass
    return bool(used >= float(max_pct))


