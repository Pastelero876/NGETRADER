from __future__ import annotations

import pandas as pd
import numpy as np

from .base import Strategy


class PairsTradingStrategy(Strategy):
    """Pairs trading por spread de log-precios con z-score.

    Recibe símbolo combinado "AAA/BBB" y utiliza precios relativos.
    Señal:
      1 si z < -th (long spread -> long AAA, short BBB)
     -1 si z > +th
    """

    def __init__(self, window: int = 60, threshold: float = 2.0) -> None:
        self.window = window
        self.threshold = threshold

    def generate_pair_signals(self, left: pd.DataFrame, right: pd.DataFrame) -> pd.Series:
        if left.empty or right.empty:
            return pd.Series(dtype=float)
        a = left["close"].astype(float).reset_index(drop=True)
        b = right["close"].astype(float).reset_index(drop=True)
        n = min(len(a), len(b))
        a = a.iloc[-n:]
        b = b.iloc[-n:]
        a_log = np.log(a.clip(lower=1e-9))
        b_log = np.log(b.clip(lower=1e-9))
        spread = a_log - b_log
        ma = spread.rolling(self.window).mean()
        std = spread.rolling(self.window).std(ddof=0).replace(0.0, 1e-9)
        z = (spread - ma) / std
        sig = pd.Series(0.0, index=z.index)
        sig[z < -self.threshold] = 1.0
        sig[z > self.threshold] = -1.0
        sig.name = "signal"
        return sig

    # Cumple interfaz base para uso genérico (no usada directamente)
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:  # type: ignore[override]
        return pd.Series(dtype=float)


