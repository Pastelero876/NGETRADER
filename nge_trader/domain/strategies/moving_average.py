from __future__ import annotations

import pandas as pd

from .base import Strategy


class MovingAverageCrossStrategy(Strategy):
    """Cruce de medias móviles simple.

    Señal:
      1 cuando SMA(fast) > SMA(slow)
     -1 cuando SMA(fast) < SMA(slow)
      0 en el resto (igualdad)
    """

    def __init__(self, fast_window: int = 10, slow_window: int = 20) -> None:
        if fast_window <= 0 or slow_window <= 0:
            raise ValueError("Las ventanas deben ser positivas")
        if fast_window >= slow_window:
            raise ValueError("fast_window debe ser menor que slow_window")
        self.fast_window = fast_window
        self.slow_window = slow_window

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:  # type: ignore[name-defined]
        if data.empty:
            return pd.Series(dtype=float)
        close = data["close"].astype(float).reset_index(drop=True)
        sma_fast = close.rolling(self.fast_window).mean()
        sma_slow = close.rolling(self.slow_window).mean()
        sig = (sma_fast > sma_slow).astype(int) - (sma_fast < sma_slow).astype(int)
        sig = sig.fillna(0.0)
        sig.name = "signal"
        return sig


