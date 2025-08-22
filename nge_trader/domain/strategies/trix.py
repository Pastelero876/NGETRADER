from __future__ import annotations

import pandas as pd

from .base import Strategy


class TRIXStrategy(Strategy):
    """TRIX (triple EMA rate of change).

    Señal:
      1 si trix > 0
     -1 si trix < 0
    """

    def __init__(self, window: int = 15) -> None:
        self.window = window

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:  # type: ignore[name-defined]
        if data.empty:
            return pd.Series(dtype=float)
        close = data["close"].astype(float).reset_index(drop=True)
        e1 = close.ewm(span=self.window, adjust=False).mean()
        e2 = e1.ewm(span=self.window, adjust=False).mean()
        e3 = e2.ewm(span=self.window, adjust=False).mean()
        trix = e3.pct_change().fillna(0.0)
        sig = pd.Series(0.0, index=close.index)
        sig[trix > 0.0] = 1.0
        sig[trix < 0.0] = -1.0
        sig.name = "signal"
        return sig


