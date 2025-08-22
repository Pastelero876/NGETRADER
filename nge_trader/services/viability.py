from __future__ import annotations

from typing import Dict, List

from nge_trader.config.settings import Settings
from nge_trader.repository.db import Database
from nge_trader.adapters.lowlat_provider import LowLatencyProvider


def get_live_costs(symbol: str) -> Dict[str, float]:
    """Obtiene costes live a partir de fuentes reales disponibles.

    - fees_bps: maker/taker desde tabla de fees_schedule (usa taker por prudencia)
    - half_spread_bps: (ask - bid) / mid * 1e4 / 2 desde L1 (LowLatencyProvider)
    """
    symu = str(symbol).upper()
    s = Settings()
    # Fees por símbolo/exchange desde DB (si no existe, default 5.0)
    fees_bps = 5.0
    try:
        exch = str(getattr(s, "market_clock_exchange", "BINANCE")).upper()
        row = Database().get_fee_schedule_any(exch, symu)
        if row:
            maker = float(row.get("maker_bps") or 0.0)
            taker = float(row.get("taker_bps") or 0.0)
            # Viabilidad conservadora: usar taker si es mayor
            fees_bps = float(max(maker, taker))
    except Exception:
        pass
    # Half-spread bps desde L1
    half_spread_bps = 0.0
    try:
        q = LowLatencyProvider().get_quote_metrics(symu)
        bid = float(q.get("bid") or 0.0)
        ask = float(q.get("ask") or 0.0)
        if bid > 0 and ask > 0:
            mid = 0.5 * (bid + ask)
            spread_bps = ((ask - bid) / max(mid, 1e-9)) * 10000.0
            half_spread_bps = 0.5 * float(spread_bps)
    except Exception:
        pass
    return {"fees_bps": float(fees_bps), "half_spread_bps": float(half_spread_bps)}


def estimate_edge_bps(symbol: str) -> float:
    """Estimación de edge (bps).

    Integración mínima: intenta leer señales recientes del feature store/DB
    si existe una columna 'edge_bps' en features, usa el último valor; si no,
    retorna una heurística simple por símbolo.
    """
    try:
        rows = Database().recent_features(symbol=str(symbol).upper(), limit=1)
        if rows:
            r = rows[0]
            v = r.get("edge_bps") or r.get("expected_edge_bps") or r.get("signal_bps")
            if v is not None:
                return float(v)
    except Exception:
        pass
    return 8.0 if str(symbol).upper() == "BTCUSDT" else 2.0


def compute_viability(symbols: List[str], buffer_bps: float = 2.0) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for s in list(symbols or []):
        c = get_live_costs(s)
        edge = float(estimate_edge_bps(s))
        ok = edge >= (float(c["fees_bps"]) + float(c["half_spread_bps"]) + float(buffer_bps))
        out[s] = {
            "edge_bps": edge,
            "fees_bps": float(c["fees_bps"]),
            "half_spread_bps": float(c["half_spread_bps"]),
            "buffer_bps": float(buffer_bps),
            "trade_viable": bool(ok),
        }
    return out


