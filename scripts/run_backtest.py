from __future__ import annotations

import argparse
import json

from nge_trader.services.backtester import BacktestSpec, run_backtest


def main() -> None:
    p = argparse.ArgumentParser(description="Run multi-symbol backtests with costs and stress")
    p.add_argument("--dataset-root", type=str, default="datasets/ohlcv")
    p.add_argument("--version", type=str, required=True)
    p.add_argument("--symbols", type=str, required=True)
    p.add_argument("--tf", type=str, default="1h")
    p.add_argument("--slip", type=float, default=10.0)
    p.add_argument("--fee", type=float, default=5.0)
    p.add_argument("--orders", type=int, default=8)
    args = p.parse_args()

    spec = BacktestSpec(
        dataset_root=args.dataset_root,
        version=args.version,
        symbols=[s.strip() for s in args.symbols.split(",") if s.strip()],
        tf=args.tf,
        slippage_bps=float(args.slip),
        fee_bps=float(args.fee),
        max_orders_per_day=int(args.orders),
    )
    res = run_backtest(spec)
    print(json.dumps(res.get("metrics", {}), ensure_ascii=False))


if __name__ == "__main__":
    main()


