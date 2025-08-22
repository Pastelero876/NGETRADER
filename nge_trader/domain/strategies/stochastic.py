from __future__ import annotations

import pandas as pd

from .base import Strategy


class StochasticOscillatorStrategy(Strategy):
    """Estrategia Stochastic %K/%D simple.

    Señal:
      1 si %K < 20
     -1 si %K > 80
    """

    def __init__(self, k_window: int = 14, d_window: int = 3) -> None:
        self.k_window = k_window
        self.d_window = d_window

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:  # type: ignore[name-defined]
        if data.empty:
            return pd.Series(dtype=float)
        high = data["high"].astype(float).reset_index(drop=True)
        low = data["low"].astype(float).reset_index(drop=True)
        close = data["close"].astype(float).reset_index(drop=True)
        lowest_low = low.rolling(self.k_window).min()
        highest_high = high.rolling(self.k_window).max()
        denom = (highest_high - lowest_low).replace(0.0, 1e-9)
        k = 100 * (close - lowest_low) / denom
        d = k.rolling(self.d_window).mean()
        _ = d  # reservado por si se usa cruce en el futuro
        sig = pd.Series(0.0, index=close.index)
        sig[k < 20.0] = 1.0
        sig[k > 80.0] = -1.0
        sig.name = "signal"
        return sig


