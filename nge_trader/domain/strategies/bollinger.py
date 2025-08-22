from __future__ import annotations

import pandas as pd

from .base import Strategy


class BollingerBandsStrategy(Strategy):
    """Estrategia de Bandas de Bollinger (mean-reversion simple).

    Señal:
      1 si close < lower_band
     -1 si close > upper_band
      0 en caso contrario
    """

    def __init__(self, window: int = 20, num_std: float = 2.0) -> None:
        self.window = window
        self.num_std = num_std

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:  # type: ignore[name-defined]
        if data.empty:
            return pd.Series(dtype=float)
        close = data["close"].astype(float).reset_index(drop=True)
        ma = close.rolling(self.window).mean()
        std = close.rolling(self.window).std(ddof=0)
        upper = ma + self.num_std * std
        lower = ma - self.num_std * std
        sig = pd.Series(0.0, index=close.index)
        sig[close < lower] = 1.0
        sig[close > upper] = -1.0
        sig.name = "signal"
        return sig


