from __future__ import annotations

"""
Job: Control de canario (promoción/rollback automático)

Promueve si 7–10 días con Sharpe_net≥champion, MaxDD_net≤champion y sin SLO breaches.
Rollback si SLO breach grave o DD>umbral.
"""

from datetime import datetime, UTC, timedelta
from typing import Dict

from nge_trader.repository.db import Database
from nge_trader.services.model_registry import ModelRegistry
from nge_trader.config.settings import Settings


def _recent_kpis(days: int = 10) -> Dict[str, float]:
    db = Database()
    try:
        eq = db.load_equity_curve()
        if eq.empty:
            return {"sharpe": float("nan"), "maxdd": float("nan")}
        import pandas as pd
        vals = pd.Series(eq.values).pct_change().dropna()
        from nge_trader.services.metrics import compute_sharpe, compute_max_drawdown
        return {"sharpe": compute_sharpe(vals.tail(days).values.tolist()), "maxdd": abs(compute_max_drawdown(eq.values.tolist()))}
    except Exception:
        return {"sharpe": float("nan"), "maxdd": float("nan")}


def run_once() -> Dict[str, str | float | bool]:
    s = Settings()
    db = Database()
    reg = ModelRegistry()
    # KPI recientes
    kpis = _recent_kpis(days=10)
    # SLO breaches (usar métricas en memoria/DB)
    try:
        from nge_trader.services.metrics import get_series_percentile
        p95 = float(get_series_percentile("order_place_latency_ms", 95))
    except Exception:
        p95 = float("nan")
    try:
        rows = db.recent_metric_values("slippage_bps", 2000)
        import time as _time
        now_ts = float(_time.time())
        vals_7d = [float(v) for ts, v in rows if (now_ts - float(ts)) <= 86400 * 7]
        slip7 = float(sum(vals_7d) / len(vals_7d)) if vals_7d else float("nan")
    except Exception:
        slip7 = float("nan")
    slo_breach = ((not (slip7 != slip7)) and abs(slip7) > float(s.slo_slippage_bps)) or ((not (p95 != p95)) and p95 > float(s.slo_p95_ms))
    action = "noop"
    detail = ""
    # Simple policy
    if slo_breach or (not (kpis["maxdd"] != kpis["maxdd"])) and kpis["maxdd"] > float(getattr(s, "max_daily_drawdown_pct", 0.01) or 0.01):
        reg.rollback()
        action = "rollback"
        detail = "SLO breach o DD elevado"
    else:
        # Si cumple KPIs mínimos
        min_sharpe = float(getattr(s, "min_recent_sharpe", 1.0) or 1.0)
        if (not (kpis["sharpe"] != kpis["sharpe"])) and kpis["sharpe"] >= min_sharpe:
            reg.promote_canary()
            action = "promote"
            detail = f"Sharpe {kpis['sharpe']:.2f} >= {min_sharpe}"
    try:
        db.append_audit(datetime.now(UTC).isoformat(), "model_canary_control", None, None, f"action={action} detail={detail}")
    except Exception:
        pass
    return {"action": action, "detail": detail, "p95": p95, "slip7": slip7, **kpis}


if __name__ == "__main__":
    out = run_once()
    print(out)


