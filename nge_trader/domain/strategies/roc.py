from __future__ import annotations

import pandas as pd

from .base import Strategy


class ROCStrategy(Strategy):
    """Rate of Change (ROC).

    Señal:
      1 si ROC > 0
     -1 si ROC < 0
    """

    def __init__(self, window: int = 12) -> None:
        self.window = window

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:  # type: ignore[name-defined]
        if data.empty:
            return pd.Series(dtype=float)
        close = data["close"].astype(float).reset_index(drop=True)
        roc = close.pct_change(self.window).fillna(0.0)
        sig = pd.Series(0.0, index=close.index)
        sig[roc > 0.0] = 1.0
        sig[roc < 0.0] = -1.0
        sig.name = "signal"
        return sig


