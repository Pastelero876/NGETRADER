from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from nge_trader.repository.db import Database
from nge_trader.services.metrics import compute_sharpe


def main() -> int:
    parser = argparse.ArgumentParser(description="Validación de performance drift (Sharpe reciente vs histórico)")
    parser.add_argument("--lookback", type=int, default=20, help="Días recientes para Sharpe reciente")
    parser.add_argument("--threshold", type=float, default=0.5, help="Desviación absoluta máxima permitida entre Sharpe reciente e histórico")
    parser.add_argument("--out", type=str, default="reports/performance_drift.json", help="Ruta de salida JSON")
    args = parser.parse_args()

    db = Database()
    eq = db.load_equity_curve()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    report = {"ok": True, "reason": None, "sharpe_recent": None, "sharpe_hist": None}
    exit_code = 0
    if eq.empty or len(eq) < max(10, int(args.lookback) + 2):
        report.update({"ok": False, "reason": "insufficient_data"})
        exit_code = 1
    else:
        rets = eq.pct_change().dropna().values.tolist()
        hist = compute_sharpe(rets)
        recent = compute_sharpe(rets[-int(args.lookback):])
        report.update({"sharpe_recent": recent, "sharpe_hist": hist})
        if pd.isna(recent) or pd.isna(hist):
            report.update({"ok": False, "reason": "nan_values"})
            exit_code = 1
        elif abs(recent - hist) > float(args.threshold):
            report.update({"ok": False, "reason": "drift_exceeds_threshold"})
            exit_code = 2
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())


