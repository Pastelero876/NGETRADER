from __future__ import annotations

import pandas as pd

from .base import Strategy


class CCIStrategy(Strategy):
    """Commodity Channel Index (CCI) simple.

    Señal:
      1 si CCI < -100
     -1 si CCI > 100
    """

    def __init__(self, window: int = 20) -> None:
        self.window = window

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:  # type: ignore[name-defined]
        if data.empty:
            return pd.Series(dtype=float)
        high = data["high"].astype(float).reset_index(drop=True)
        low = data["low"].astype(float).reset_index(drop=True)
        close = data["close"].astype(float).reset_index(drop=True)
        typical_price = (high + low + close) / 3.0
        sma = typical_price.rolling(self.window).mean()
        mad = (typical_price - sma).abs().rolling(self.window).mean().replace(0.0, 1e-9)
        cci = (typical_price - sma) / (0.015 * mad)
        sig = pd.Series(0.0, index=close.index)
        sig[cci < -100.0] = 1.0
        sig[cci > 100.0] = -1.0
        sig.name = "signal"
        return sig


