from __future__ import annotations

import pandas as pd

from .base import Strategy


class MACDStrategy(Strategy):
    """Señal por cruce MACD (EMA fast - EMA slow vs signal).

    Señal:
      1 si macd > signal
     -1 si macd < signal
      0 si iguales
    """

    def __init__(self, fast: int = 12, slow: int = 26, signal_win: int = 9) -> None:
        self.fast = fast
        self.slow = slow
        self.signal_win = signal_win

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:  # type: ignore[name-defined]
        if data.empty:
            return pd.Series(dtype=float)
        close = data["close"].astype(float).reset_index(drop=True)
        ema_fast = close.ewm(span=self.fast, adjust=False).mean()
        ema_slow = close.ewm(span=self.slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=self.signal_win, adjust=False).mean()
        sig = (macd > signal).astype(float) - (macd < signal).astype(float)
        sig.name = "signal"
        return sig


