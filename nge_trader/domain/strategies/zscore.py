from __future__ import annotations

import pandas as pd

from .base import Strategy


class ZScoreReversionStrategy(Strategy):
    """Reversión a la media por Z-Score de los retornos o de la serie.

    Señal:
      1 si z < -threshold
     -1 si z > +threshold
    """

    def __init__(self, window: int = 20, threshold: float = 1.5) -> None:
        self.window = window
        self.threshold = threshold

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:  # type: ignore[name-defined]
        if data.empty:
            return pd.Series(dtype=float)
        close = data["close"].astype(float).reset_index(drop=True)
        ma = close.rolling(self.window).mean()
        std = close.rolling(self.window).std(ddof=0).replace(0.0, 1e-9)
        z = (close - ma) / std
        sig = pd.Series(0.0, index=close.index)
        sig[z < -self.threshold] = 1.0
        sig[z > self.threshold] = -1.0
        sig.name = "signal"
        return sig


