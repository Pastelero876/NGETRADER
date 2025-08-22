from __future__ import annotations

import pandas as pd

from .base import Strategy


class KeltnerChannelsStrategy(Strategy):
    """Keltner Channels (EMA +/- mult * ATR).

    Señal:
      1 si close < lower_channel
     -1 si close > upper_channel
    """

    def __init__(self, ema_window: int = 20, atr_window: int = 10, atr_mult: float = 2.0) -> None:
        self.ema_window = ema_window
        self.atr_window = atr_window
        self.atr_mult = atr_mult

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:  # type: ignore[name-defined]
        if data.empty:
            return pd.Series(dtype=float)
        high = data["high"].astype(float).reset_index(drop=True)
        low = data["low"].astype(float).reset_index(drop=True)
        close = data["close"].astype(float).reset_index(drop=True)
        ema = close.ewm(span=self.ema_window, adjust=False).mean()
        prev_close = close.shift(1)
        tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        atr = tr.rolling(self.atr_window).mean()
        upper = ema + self.atr_mult * atr
        lower = ema - self.atr_mult * atr
        sig = pd.Series(0.0, index=close.index)
        sig[close < lower] = 1.0
        sig[close > upper] = -1.0
        sig.name = "signal"
        return sig


