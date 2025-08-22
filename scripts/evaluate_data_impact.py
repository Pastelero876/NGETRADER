from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import pandas as pd

from nge_trader.services.app_service import AppService
from nge_trader.services.backtester import SignalBacktester
from nge_trader.domain.strategies.moving_average import MovingAverageCrossStrategy
from nge_trader.domain.strategies.rsi import RSIStrategy
from nge_trader.repository.db import Database
from nge_trader.services.metrics import (
    compute_sharpe,
    compute_sortino,
    compute_max_drawdown,
    compute_win_rate,
    compute_profit_factor,
)


def build_strategy(key: str):
    if key == "ma_cross":
        return MovingAverageCrossStrategy(10, 20)
    if key == "rsi":
        return RSIStrategy(14, 30.0, 70.0)
    return MovingAverageCrossStrategy(10, 20)


def equity_to_returns(eq: pd.Series) -> pd.Series:
    return eq.astype(float).pct_change().dropna() if not eq.empty else pd.Series(dtype=float)


def summarize_performance(eq: pd.Series) -> Dict[str, float]:
    rets = equity_to_returns(eq)
    if rets.empty:
        return {"sharpe": float("nan"), "sortino": float("nan"), "mdd": float("nan"), "win_rate": float("nan"), "profit_factor": float("nan")}
    return {
        "sharpe": compute_sharpe(rets.values.tolist()),
        "sortino": compute_sortino(rets.values.tolist()),
        "mdd": compute_max_drawdown(eq.values.tolist()),
        "win_rate": compute_win_rate(rets.values.tolist()),
        "profit_factor": compute_profit_factor(rets.values.tolist()),
    }


def apply_sentiment_bias(sig: pd.Series, symbol: str, window_min: int = 180) -> pd.Series:
    db = Database()
    try:
        s = db.aggregated_sentiment(symbol=symbol, minutes=int(window_min))
        if s > 0.2:
            # sesgo positivo: empujar señales a favor
            return (sig + 0.2).clip(-1.0, 1.0)
        if s < -0.2:
            # sesgo negativo: reducir longs
            return (sig - 0.2).clip(-1.0, 1.0)
    except Exception:
        pass
    return sig


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluar impacto de datos alternativos (sentimiento) en performance")
    parser.add_argument("--symbols", type=str, required=True, help="Símbolos separados por coma")
    parser.add_argument("--strategy", type=str, default="ma_cross", help="Estrategia base: ma_cross|rsi")
    parser.add_argument("--out", type=str, default="reports/data_impact.json")
    args = parser.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    svc = AppService()
    out: Dict[str, Dict] = {"symbols": {}, "summary": {"improved": 0, "degraded": 0}}

    for sym in [s.strip().upper() for s in args.symbols.split(",") if s.strip()]:
        try:
            df = svc.data_provider.get_daily_adjusted(sym)
            strat = build_strategy(args.strategy)
            sig = strat.generate_signals(df)
            bt = SignalBacktester()
            res_base = bt.run(df, sig)
            eq_base = res_base.get("equity_curve") if isinstance(res_base.get("equity_curve"), pd.Series) else pd.Series(dtype=float)
            perf_base = summarize_performance(eq_base)

            sig_bias = apply_sentiment_bias(sig, sym)
            res_bias = bt.run(df, sig_bias)
            eq_bias = res_bias.get("equity_curve") if isinstance(res_bias.get("equity_curve"), pd.Series) else pd.Series(dtype=float)
            perf_bias = summarize_performance(eq_bias)

            delta_sharpe = (perf_bias["sharpe"] - perf_base["sharpe"]) if perf_base["sharpe"] == perf_base["sharpe"] else float("nan")
            improved = (delta_sharpe == delta_sharpe) and (delta_sharpe > 0)  # not nan and positive
            out["symbols"][sym] = {
                "base": perf_base,
                "bias": perf_bias,
                "delta_sharpe": delta_sharpe,
            }
            out["summary"]["improved"] += 1 if improved else 0
            out["summary"]["degraded"] += 1 if ((delta_sharpe == delta_sharpe) and (delta_sharpe < 0)) else 0
        except Exception as exc:  # noqa: BLE001
            out["symbols"][sym] = {"error": str(exc)}

    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


