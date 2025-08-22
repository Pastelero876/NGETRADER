from __future__ import annotations

import pandas as pd


def compute_atr(data: pd.DataFrame, window: int = 14) -> pd.Series:
    """Calcula el ATR clásico (Average True Range)."""
    high = data["high"].astype(float)
    low = data["low"].astype(float)
    close = data["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(window).mean()
    atr.name = "atr"
    return atr


def position_size_by_risk(equity: float, risk_pct: float, stop_distance: float, price: float) -> float:
    """Calcula tamaño de posición dado un riesgo porcentual y distancia de stop."""
    if price <= 0 or stop_distance <= 0 or risk_pct <= 0 or equity <= 0:
        return 0.0
    risk_amount = equity * risk_pct
    per_unit_risk = stop_distance
    qty = risk_amount / per_unit_risk
    # Convertir a unidades del activo
    return max(qty, 0.0)


