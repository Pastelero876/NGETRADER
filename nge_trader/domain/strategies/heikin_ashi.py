from __future__ import annotations

import pandas as pd

from .base import Strategy


class HeikinAshiTrendStrategy(Strategy):
    """Tendencia basada en velas Heikin Ashi.

    Señal:
      1 si HA_close > HA_open
     -1 si HA_close < HA_open
    """

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:  # type: ignore[name-defined]
        if data.empty:
            return pd.Series(dtype=float)
        open_ = data.get("open")
        high = data.get("high")
        low = data.get("low")
        close = data.get("close")
        if open_ is None or high is None or low is None or close is None:
            return pd.Series(dtype=float)
        o = pd.Series(open_).astype(float).reset_index(drop=True)
        h = pd.Series(high).astype(float).reset_index(drop=True)
        l = pd.Series(low).astype(float).reset_index(drop=True)
        c = pd.Series(close).astype(float).reset_index(drop=True)

        ha_close = (o + h + l + c) / 4.0
        ha_open = pd.Series(index=ha_close.index, dtype=float)
        ha_open.iloc[0] = o.iloc[0]
        for i in range(1, len(ha_close)):
            ha_open.iloc[i] = (ha_open.iloc[i - 1] + ha_close.iloc[i - 1]) / 2.0
        sig = pd.Series(0.0, index=ha_close.index)
        sig[ha_close > ha_open] = 1.0
        sig[ha_close < ha_open] = -1.0
        sig.name = "signal"
        return sig


