from __future__ import annotations

import pandas as pd

from .base import Strategy


class VWAPDeviationStrategy(Strategy):
    """Desviación sobre/under VWAP intradiario (aprox con daily OHLCV acumulado).

    Señal:
      1 si close < vwap * (1 - dev)
     -1 si close > vwap * (1 + dev)
    """

    def __init__(self, deviation: float = 0.01) -> None:
        self.deviation = deviation

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:  # type: ignore[name-defined]
        if data.empty:
            return pd.Series(dtype=float)
        close = data["close"].astype(float).reset_index(drop=True)
        vol = pd.Series(data.get("volume", 0)).astype(float).reset_index(drop=True)
        # VWAP aproximado en datos diarios
        typical = close  # si no tenemos intradía, usamos close como aproximación
        cum_pv = (typical * vol).cumsum().replace(0.0, 1e-9)
        cum_v = vol.cumsum().replace(0.0, 1e-9)
        vwap = (cum_pv / cum_v).fillna(method="ffill").fillna(typical)
        upper = vwap * (1.0 + self.deviation)
        lower = vwap * (1.0 - self.deviation)
        sig = pd.Series(0.0, index=close.index)
        sig[close < lower] = 1.0
        sig[close > upper] = -1.0
        sig.name = "signal"
        return sig


