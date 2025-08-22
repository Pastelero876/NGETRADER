from __future__ import annotations

import pandas as pd

from .base import Strategy


class MomentumStrategy(Strategy):
    """Estrategia de Momentum por retorno N-periodos.

    Señal:
      1 si retorno(window) > 0
     -1 si retorno(window) < 0
    """

    def __init__(self, window: int = 20) -> None:
        self.window = window

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:  # type: ignore[name-defined]
        if data.empty:
            return pd.Series(dtype=float)
        close = data["close"].astype(float).reset_index(drop=True)
        ret = close.pct_change(self.window)
        sig = pd.Series(0.0, index=close.index)
        sig[ret > 0.0] = 1.0
        sig[ret < 0.0] = -1.0
        sig.name = "signal"
        return sig


