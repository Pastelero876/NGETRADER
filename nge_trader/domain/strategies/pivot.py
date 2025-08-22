from __future__ import annotations

import pandas as pd

from .base import Strategy


class PivotReversalStrategy(Strategy):
    """Reversals basados en Pivots clásicos (H+L+C)/3.

    Señal:
      1 si close > pivot y > close[-1]
     -1 si close < pivot y < close[-1]
    """

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:  # type: ignore[name-defined]
        if data.empty:
            return pd.Series(dtype=float)
        high = data["high"].astype(float).reset_index(drop=True)
        low = data["low"].astype(float).reset_index(drop=True)
        close = data["close"].astype(float).reset_index(drop=True)
        pivot = (high.shift(1) + low.shift(1) + close.shift(1)) / 3.0
        sig = pd.Series(0.0, index=close.index)
        sig[(close > pivot) & (close > close.shift(1))] = 1.0
        sig[(close < pivot) & (close < close.shift(1))] = -1.0
        sig.name = "signal"
        return sig


