from __future__ import annotations

import pandas as pd

from .base import Strategy


class EMACrossoverStrategy(Strategy):
    """Cruce de EMAs (rápida vs lenta).

    Señal:
      1 si ema_fast > ema_slow
     -1 si ema_fast < ema_slow
    """

    def __init__(self, fast: int = 9, slow: int = 21) -> None:
        self.fast = fast
        self.slow = slow

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:  # type: ignore[name-defined]
        if data.empty:
            return pd.Series(dtype=float)
        close = data["close"].astype(float).reset_index(drop=True)
        ema_fast = close.ewm(span=self.fast, adjust=False).mean()
        ema_slow = close.ewm(span=self.slow, adjust=False).mean()
        sig = (ema_fast > ema_slow).astype(float) - (ema_fast < ema_slow).astype(float)
        sig.name = "signal"
        return sig


