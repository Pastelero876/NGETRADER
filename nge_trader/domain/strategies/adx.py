from __future__ import annotations

import pandas as pd

from .base import Strategy


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr


class ADXStrategy(Strategy):
    """ADX + DI para confirmar tendencia.

    Señal:
      1 si +DI > -DI y ADX > threshold
     -1 si -DI > +DI y ADX > threshold
      0 en caso contrario
    """

    def __init__(self, window: int = 14, adx_threshold: float = 20.0) -> None:
        self.window = window
        self.adx_threshold = adx_threshold

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:  # type: ignore[name-defined]
        if data.empty:
            return pd.Series(dtype=float)
        high = data["high"].astype(float).reset_index(drop=True)
        low = data["low"].astype(float).reset_index(drop=True)
        close = data["close"].astype(float).reset_index(drop=True)

        up_move = high.diff()
        down_move = -low.diff()
        plus_dm = ((up_move > down_move) & (up_move > 0)).astype(float) * up_move.clip(lower=0.0)
        minus_dm = ((down_move > up_move) & (down_move > 0)).astype(float) * down_move.clip(lower=0.0)

        tr = _true_range(high, low, close)
        alpha = 1.0 / float(self.window)
        atr = tr.ewm(alpha=alpha, adjust=False).mean().replace(0.0, 1e-9)
        plus_di = 100.0 * plus_dm.ewm(alpha=alpha, adjust=False).mean() / atr
        minus_di = 100.0 * minus_dm.ewm(alpha=alpha, adjust=False).mean() / atr

        dx = (100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, 1e-9)).fillna(0.0)
        adx = dx.ewm(alpha=alpha, adjust=False).mean()

        sig = pd.Series(0.0, index=close.index)
        sig[(plus_di > minus_di) & (adx > self.adx_threshold)] = 1.0
        sig[(minus_di > plus_di) & (adx > self.adx_threshold)] = -1.0
        sig.name = "signal"
        return sig


