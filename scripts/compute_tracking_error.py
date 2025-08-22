from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from nge_trader.services.app_service import AppService
from nge_trader.services.backtester import SignalBacktester, SimpleBuyHoldBacktester
from nge_trader.domain.strategies.moving_average import MovingAverageCrossStrategy
from nge_trader.domain.strategies.rsi import RSIStrategy
from nge_trader.repository.db import Database


def build_strategy(key: str):
    if key == "ma_cross":
        return MovingAverageCrossStrategy(10, 20)
    if key == "rsi":
        return RSIStrategy(14, 30.0, 70.0)
    # fallback: buy&hold as benchmark
    return None


def compute_metrics(live_eq: pd.Series, paper_eq: pd.Series) -> dict:
    # Alinear por índice/fecha
    df = pd.DataFrame({"live": live_eq, "paper": paper_eq}).dropna()
    if df.empty or len(df) < 5:
        return {"ok": False, "reason": "insufficient_data"}
    live_ret = df["live"].pct_change().dropna()
    paper_ret = df["paper"].pct_change().dropna()
    n = min(len(live_ret), len(paper_ret))
    live_ret = live_ret.iloc[-n:]
    paper_ret = paper_ret.iloc[-n:]
    diff = live_ret.values - paper_ret.values
    te_std = float(np.std(diff))
    te_mae = float(np.mean(np.abs(diff)))
    corr = float(np.corrcoef(live_ret.values, paper_ret.values)[0, 1]) if n > 1 else float("nan")
    return {
        "ok": True,
        "tracking_error_std": te_std,
        "tracking_error_mae": te_mae,
        "correlation": corr,
        "n": int(n),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute tracking error between live equity and paper backtest")
    parser.add_argument("--symbol", type=str, required=True)
    parser.add_argument("--strategy", type=str, default="ma_cross")
    parser.add_argument("--out", type=str, default="reports/tracking_error.json")
    parser.add_argument("--max_te_std", type=float, default=0.05)
    args = parser.parse_args()

    svc = AppService()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    # Live equity desde DB
    db = Database()
    live_eq = db.load_equity_curve()
    # Paper: backtest simple usando estrategia indicada
    df = svc.data_provider.get_daily_adjusted(args.symbol)
    strat = build_strategy(args.strategy)
    if strat is not None and hasattr(strat, "generate_signals"):
        sig = strat.generate_signals(df)
        backtester = SignalBacktester()
        res = backtester.run(df, sig)
        paper_eq = res.get("equity_curve") if isinstance(res.get("equity_curve"), pd.Series) else pd.Series(dtype=float)
    else:
        bh = SimpleBuyHoldBacktester()
        res = bh.run(df)
        paper_eq = res.get("equity_curve") if isinstance(res.get("equity_curve"), pd.Series) else pd.Series(dtype=float)

    metrics = compute_metrics(live_eq, paper_eq)
    Path(args.out).write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    if metrics.get("ok") and float(metrics.get("tracking_error_std") or 0.0) <= float(args.max_te_std):
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())


