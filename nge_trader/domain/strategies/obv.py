from __future__ import annotations

import pandas as pd

from .base import Strategy


class OBVStrategy(Strategy):
    """On-Balance Volume (OBV) con señal por pendiente.

    Señal:
      1 si OBV creciente (dif > 0)
     -1 si OBV decreciente (dif < 0)
    """

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:  # type: ignore[name-defined]
        if data.empty:
            return pd.Series(dtype=float)
        close = data["close"].astype(float).reset_index(drop=True)
        vol = data.get("volume")
        if vol is None:
            return pd.Series(0.0, index=close.index)
        vol = pd.Series(vol).astype(float).reset_index(drop=True)
        direction = (close.diff().fillna(0.0) > 0).astype(int) - (close.diff().fillna(0.0) < 0).astype(int)
        obv = (direction * vol).cumsum()
        dif = obv.diff().fillna(0.0)
        sig = pd.Series(0.0, index=close.index)
        sig[dif > 0] = 1.0
        sig[dif < 0] = -1.0
        sig.name = "signal"
        return sig


