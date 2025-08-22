from __future__ import annotations

import pandas as pd

from .base import Strategy


class RSIStrategy(Strategy):
    """Estrategia simple RSI: compra si RSI<30, vende si RSI>70."""

    def __init__(self, window: int = 14, low: float = 30.0, high: float = 70.0) -> None:
        self.window = window
        self.low = low
        self.high = high

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:  # type: ignore[name-defined]
        if data.empty:
            return pd.Series(dtype=float)
        close = data["close"].astype(float).reset_index(drop=True)
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)
        avg_gain = gain.rolling(self.window).mean()
        avg_loss = loss.rolling(self.window).mean()
        rs = avg_gain / (avg_loss.replace(0.0, 1e-9))
        rsi = 100 - (100 / (1 + rs))
        sig = pd.Series(0.0, index=close.index)
        sig[rsi < self.low] = 1.0
        sig[rsi > self.high] = -1.0
        sig.name = "signal"
        return sig


