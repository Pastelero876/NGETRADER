from __future__ import annotations

import pandas as pd

from .base import Strategy


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr


class SupertrendStrategy(Strategy):
    """Supertrend simplificado basado en ATR.

    Señal:
      1 si close > basic_upper_band
     -1 si close < basic_lower_band
      0 en caso contrario
    """

    def __init__(self, atr_window: int = 10, multiplier: float = 3.0) -> None:
        self.atr_window = atr_window
        self.multiplier = multiplier

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:  # type: ignore[name-defined]
        if data.empty:
            return pd.Series(dtype=float)
        high = data["high"].astype(float).reset_index(drop=True)
        low = data["low"].astype(float).reset_index(drop=True)
        close = data["close"].astype(float).reset_index(drop=True)

        tr = _true_range(high, low, close)
        atr = tr.rolling(self.atr_window).mean()
        hl2 = (high + low) / 2.0
        upper = hl2 + self.multiplier * atr
        lower = hl2 - self.multiplier * atr

        sig = pd.Series(0.0, index=close.index)
        sig[close > upper] = 1.0
        sig[close < lower] = -1.0
        sig.name = "signal"
        return sig


