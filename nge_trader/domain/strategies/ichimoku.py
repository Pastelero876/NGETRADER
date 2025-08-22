from __future__ import annotations

import pandas as pd

from .base import Strategy


class IchimokuStrategy(Strategy):
    """Ichimoku Kinko Hyo simplificado.

    Señal:
      1 si close > cloud_top y Tenkan > Kijun
     -1 si close < cloud_bottom y Tenkan < Kijun
    """

    def __init__(self, conv: int = 9, base: int = 26, span_b: int = 52) -> None:
        self.conv = conv
        self.base = base
        self.span_b = span_b

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:  # type: ignore[name-defined]
        if data.empty:
            return pd.Series(dtype=float)
        high = data["high"].astype(float).reset_index(drop=True)
        low = data["low"].astype(float).reset_index(drop=True)
        close = data["close"].astype(float).reset_index(drop=True)

        tenkan = (high.rolling(self.conv).max() + low.rolling(self.conv).min()) / 2.0
        kijun = (high.rolling(self.base).max() + low.rolling(self.base).min()) / 2.0
        span_a = ((tenkan + kijun) / 2.0).shift(self.base)
        span_b = ((high.rolling(self.span_b).max() + low.rolling(self.span_b).min()) / 2.0).shift(self.base)

        cloud_top = pd.concat([span_a, span_b], axis=1).max(axis=1)
        cloud_bottom = pd.concat([span_a, span_b], axis=1).min(axis=1)

        sig = pd.Series(0.0, index=close.index)
        sig[(close > cloud_top) & (tenkan > kijun)] = 1.0
        sig[(close < cloud_bottom) & (tenkan < kijun)] = -1.0
        sig.name = "signal"
        return sig


