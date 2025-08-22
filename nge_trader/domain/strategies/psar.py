from __future__ import annotations

import pandas as pd

from .base import Strategy


class ParabolicSARStrategy(Strategy):
    """Parabolic SAR aproximado.

    Señal direccional según tendencia estimada por PSAR.
    """

    def __init__(self, step: float = 0.02, max_step: float = 0.2) -> None:
        self.step = step
        self.max_step = max_step

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:  # type: ignore[name-defined]
        if data.empty:
            return pd.Series(dtype=float)
        high = data["high"].astype(float).reset_index(drop=True)
        low = data["low"].astype(float).reset_index(drop=True)
        close = data["close"].astype(float).reset_index(drop=True)

        psar = close.copy()
        bull = True
        af = self.step
        ep = high.iloc[0]
        sar = low.iloc[0]
        for i in range(2, len(close)):
            prev_sar = sar
            if bull:
                sar = prev_sar + af * (ep - prev_sar)
                sar = min(sar, low.iloc[i - 1], low.iloc[i - 2])
                if high.iloc[i] > ep:
                    ep = high.iloc[i]
                    af = min(af + self.step, self.max_step)
                if low.iloc[i] < sar:
                    bull = False
                    sar = ep
                    ep = low.iloc[i]
                    af = self.step
            else:
                sar = prev_sar + af * (ep - prev_sar)
                sar = max(sar, high.iloc[i - 1], high.iloc[i - 2])
                if low.iloc[i] < ep:
                    ep = low.iloc[i]
                    af = min(af + self.step, self.max_step)
                if high.iloc[i] > sar:
                    bull = True
                    sar = ep
                    ep = high.iloc[i]
                    af = self.step
            psar.iloc[i] = sar

        sig = pd.Series(0.0, index=close.index)
        sig[close > psar] = 1.0
        sig[close < psar] = -1.0
        sig.name = "signal"
        return sig


