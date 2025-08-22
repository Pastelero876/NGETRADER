from __future__ import annotations

import pandas as pd

from .base import Strategy


class ChaikinMoneyFlowStrategy(Strategy):
    """Chaikin Money Flow (CMF) con ventana.

    Señal:
      1 si CMF > 0
     -1 si CMF < 0
    """

    def __init__(self, window: int = 20) -> None:
        self.window = window

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:  # type: ignore[name-defined]
        if data.empty:
            return pd.Series(dtype=float)
        high = data["high"].astype(float).reset_index(drop=True)
        low = data["low"].astype(float).reset_index(drop=True)
        close = data["close"].astype(float).reset_index(drop=True)
        vol = pd.Series(data.get("volume", 0)).astype(float).reset_index(drop=True)

        mfm = ((close - low) - (high - close)) / (high - low).replace(0.0, 1e-9)
        mfv = mfm * vol
        cmf = mfv.rolling(self.window).sum() / vol.rolling(self.window).sum().replace(0.0, 1e-9)

        sig = pd.Series(0.0, index=close.index)
        sig[cmf > 0.0] = 1.0
        sig[cmf < 0.0] = -1.0
        sig.name = "signal"
        return sig


