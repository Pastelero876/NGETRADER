from __future__ import annotations

import pandas as pd

from .base import Strategy


class DonchianBreakoutStrategy(Strategy):
    """Donchian Channel breakout.

    Señal:
      1 si close > upper_channel
     -1 si close < lower_channel
    """

    def __init__(self, window: int = 20) -> None:
        self.window = window

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:  # type: ignore[name-defined]
        if data.empty:
            return pd.Series(dtype=float)
        high = data["high"].astype(float).reset_index(drop=True)
        low = data["low"].astype(float).reset_index(drop=True)
        close = data["close"].astype(float).reset_index(drop=True)
        upper = high.rolling(self.window).max()
        lower = low.rolling(self.window).min()
        sig = pd.Series(0.0, index=close.index)
        sig[close > upper] = 1.0
        sig[close < lower] = -1.0
        sig.name = "signal"
        return sig


