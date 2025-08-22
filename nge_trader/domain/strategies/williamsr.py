from __future__ import annotations

import pandas as pd

from .base import Strategy


class WilliamsRStrategy(Strategy):
    """Williams %R.

    Señal:
      1 si %R < -80 (sobreventa)
     -1 si %R > -20 (sobrecompra)
    """

    def __init__(self, window: int = 14) -> None:
        self.window = window

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:  # type: ignore[name-defined]
        if data.empty:
            return pd.Series(dtype=float)
        high = data["high"].astype(float).reset_index(drop=True)
        low = data["low"].astype(float).reset_index(drop=True)
        close = data["close"].astype(float).reset_index(drop=True)
        highest_high = high.rolling(self.window).max()
        lowest_low = low.rolling(self.window).min()
        denom = (highest_high - lowest_low).replace(0.0, 1e-9)
        wr = -100.0 * (highest_high - close) / denom
        sig = pd.Series(0.0, index=close.index)
        sig[wr < -80.0] = 1.0
        sig[wr > -20.0] = -1.0
        sig.name = "signal"
        return sig


