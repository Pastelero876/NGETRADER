from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from nge_trader.services.app_service import AppService
from nge_trader.services.backtester import SignalBacktester, SimpleBuyHoldBacktester, BacktestResult
from nge_trader.services.metrics import (
    compute_max_drawdown,
    compute_sharpe,
    compute_sortino,
    compute_win_rate,
)
from nge_trader.domain.strategies.moving_average import MovingAverageCrossStrategy
from nge_trader.domain.strategies.rsi import RSIStrategy


def _build_strategy(strategy_key: str, params: Dict[str, Any]):
    if strategy_key == "ma_cross":
        return MovingAverageCrossStrategy(
            int(params.get("fast_window", 10)), int(params.get("slow_window", 20))
        )
    if strategy_key == "rsi":
        return RSIStrategy(
            int(params.get("window", 14)), float(params.get("low", 30.0)), float(params.get("high", 70.0))
        )
    return None


def _param_combinations(param_grid: Dict[str, Iterable[Any]]) -> List[Dict[str, Any]]:
    if not param_grid:
        return [{}]
    keys = list(param_grid.keys())
    values = [list(param_grid[k]) for k in keys]
    combos: List[Dict[str, Any]] = []
    for vals in product(*values):
        combos.append({k: v for k, v in zip(keys, vals)})
    return combos


def _compute_metrics(result: BacktestResult) -> Dict[str, float]:
    if result.equity_curve.empty:
        return {"total_return": float("nan"), "sharpe": float("nan"), "sortino": float("nan"), "max_drawdown": float("nan"), "win_rate": float("nan")}
    eq = result.equity_curve
    rets = eq.pct_change().fillna(0.0)
    return {
        "total_return": float(eq.iloc[-1] / eq.iloc[0] - 1.0),
        "sharpe": float(compute_sharpe(rets)),
        "sortino": float(compute_sortino(rets)),
        "max_drawdown": float(compute_max_drawdown(eq.values)),
        "win_rate": float(compute_win_rate(rets)),
    }


@dataclass
class SweepResult:
    rows: pd.DataFrame
    out_dir: Path


def run_param_sweep(
    symbols: List[str],
    strategy_key: str,
    param_grid: Dict[str, Iterable[Any]] | None = None,
    date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
) -> SweepResult:
    service = AppService()
    combos = _param_combinations(param_grid or {})
    records: List[Dict[str, Any]] = []
    for symbol in symbols:
        df = service.data_provider.get_daily_adjusted(symbol)
        for params in combos:
            if strategy_key == "buyhold":
                backtester = SimpleBuyHoldBacktester()
                result = backtester.run(df, date_range=date_range)
            else:
                strat = _build_strategy(strategy_key, params)
                if strat is None:
                    continue
                signal = strat.generate_signals(df)
                backtester = SignalBacktester()
                result = backtester.run(df, signal, date_range=date_range)
            metrics = _compute_metrics(result)
            row: Dict[str, Any] = {
                "symbol": symbol,
                "strategy": strategy_key,
                "params": params,
                **metrics,
            }
            records.append(row)

    rows = pd.DataFrame(records)
    # Ordenar por retorno total y sharpe
    if not rows.empty:
        rows = rows.sort_values(["total_return", "sharpe"], ascending=[False, False])
    out_dir = Path("data/sweeps") / pd.Timestamp.now(tz=pd.Timestamp.utcnow().tz).strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(out_dir / "results.csv", index=False)
    return SweepResult(rows=rows, out_dir=out_dir)


