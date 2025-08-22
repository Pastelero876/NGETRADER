from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from nge_trader.services.analytics import (
    make_report_dir,
    export_equity_csv,
    export_trades_csv,
    export_tearsheet,
    export_report_html,
)
from nge_trader.repository.db import Database


def _load_equity_series(db: Database) -> pd.Series:
    vals = db.recent_metric_series("equity", 10000)
    if not vals:
        return pd.Series(dtype=float)
    df = pd.DataFrame(vals, columns=["ts", "value"]).sort_values("ts")
    s = pd.Series(df["value"].astype(float).values)
    return s


def _load_trades(db: Database) -> list[dict]:
    # Simplificación: recent_fills como trades
    fills = db.recent_fills(1000)
    return list(fills)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default=None, help="Directorio de salida del reporte")
    args = parser.parse_args()

    out_dir = Path(args.out) if args.out else make_report_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    db = Database()
    equity = _load_equity_series(db)
    trades = _load_trades(db)
    # Exports
    export_equity_csv(out_dir, equity)
    export_trades_csv(out_dir, trades)
    export_tearsheet(out_dir, equity)
    # Métricas clave
    metrics = {}
    if not equity.empty and equity.iloc[0] != 0:
        ret = equity.pct_change().fillna(0.0)
        metrics["total_return"] = float((equity.iloc[-1] / equity.iloc[0]) - 1.0)
        vol = float(ret.std(ddof=0)) or 1e-9
        metrics["sharpe"] = float(ret.mean() / vol)
        peak = equity.cummax().replace(0.0, 1e-9)
        dd = (equity - peak) / peak
        metrics["max_drawdown"] = float(dd.min())
    export_report_html(out_dir, metrics)
    print({"status": "ok", "dir": str(out_dir)})


if __name__ == "__main__":
    main()


