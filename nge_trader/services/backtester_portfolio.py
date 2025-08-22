from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd

from .portfolio import PortfolioEngine


@dataclass
class PortfolioBacktestResult:
    equity_curve: pd.Series
    weights_history: list[tuple[pd.Timestamp, dict[str, float]]]
    stats: dict[str, float]


def run_portfolio_backtest(
    data_by_symbol: dict[str, pd.DataFrame],
    method: str = "risk_parity",
    window: int = 60,
    rebalance_days: int = 21,
    initial_capital: float = 10_000.0,
    date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
) -> PortfolioBacktestResult:
    # Construir matriz de precios y retornos
    closes: dict[str, pd.Series] = {}
    for sym, df in data_by_symbol.items():
        if df.empty:
            continue
        s = df[["date", "close"]].copy() if "date" in df.columns else df[["close"]].copy()
        if "date" in s.columns:
            s["date"] = pd.to_datetime(s["date"])  # type: ignore[assignment]
            s = s.set_index("date").sort_index()
        s = s["close"].astype(float)
        closes[sym] = s
    if not closes:
        return PortfolioBacktestResult(pd.Series(dtype=float), [], {})
    prices = pd.DataFrame(closes).dropna()
    if date_range is not None:
        start, end = date_range
        prices = prices[(prices.index >= start) & (prices.index <= end)]
    if prices.empty:
        return PortfolioBacktestResult(pd.Series(dtype=float), [], {})
    returns = prices.pct_change().dropna()

    engine = PortfolioEngine()
    equity = initial_capital
    eq_curve: list[float] = []
    weights_history: list[tuple[pd.Timestamp, dict[str, float]]] = []
    current_weights = {c: 1.0 / len(prices.columns) for c in prices.columns}
    last_reb_idx = None

    for i in range(len(returns)):
        idx = returns.index[i]
        # rebalanceo periódico
        if (last_reb_idx is None) or ((i - last_reb_idx) >= max(rebalance_days, 1)):
            ret_window = returns.iloc[max(0, i - window + 1) : i + 1]
            if method == "min_variance":
                alloc = engine.min_variance(ret_window) if not ret_window.empty else None
            elif method == "black_litterman":
                alloc = engine.black_litterman_basic(ret_window) if not ret_window.empty else None
            else:
                alloc = engine.risk_parity(ret_window) if not ret_window.empty else None
            if alloc and alloc.weights:
                current_weights = alloc.weights
            weights_history.append((idx, dict(current_weights)))
            last_reb_idx = i
        # actualizar equity por retorno de cartera
        day_ret = float((returns.iloc[i] * pd.Series(current_weights)).sum())
        equity *= (1.0 + day_ret)
        eq_curve.append(equity)

    eq_series = pd.Series(eq_curve, index=returns.index)
    ret_series = eq_series.pct_change().fillna(0.0)
    stats = {
        "total_return": float(eq_series.iloc[-1] / eq_series.iloc[0] - 1.0) if not eq_series.empty else 0.0,
        "volatility": float(ret_series.std(ddof=0) * (252 ** 0.5)) if not ret_series.empty else 0.0,
    }
    return PortfolioBacktestResult(eq_series, weights_history, stats)


