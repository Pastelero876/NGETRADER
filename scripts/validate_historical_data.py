from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from nge_trader.services.app_service import AppService


def detect_gaps(df: pd.DataFrame) -> List[Tuple[str, str]]:
    gaps: List[Tuple[str, str]] = []
    if df.empty or "date" not in df.columns:
        return gaps
    d = pd.to_datetime(df["date"]).sort_values().reset_index(drop=True)
    diffs = d.diff().dropna()
    # marcar gaps > 1 día
    idxs = diffs[diffs > pd.Timedelta(days=1)].index
    for i in idxs:
        gaps.append((str(d[i - 1].date()), str(d[i].date())))
    return gaps


def detect_outliers(df: pd.DataFrame, z_thresh: float = 8.0) -> List[Tuple[str, float]]:
    if df.empty or "close" not in df.columns or "date" not in df.columns:
        return []
    s = df["close"].astype(float).reset_index(drop=True)
    ret = s.pct_change().dropna()
    if ret.empty:
        return []
    z = (ret - ret.mean()) / (ret.std(ddof=0) or 1e-9)
    bad = z[np.abs(z) > float(z_thresh)]
    out: List[Tuple[str, float]] = []
    for i in bad.index.tolist():
        dt = str(df["date"].iloc[int(i)])
        out.append((dt, float(ret.iloc[int(i)])))
    return out


def detect_bad_ticks(df: pd.DataFrame) -> List[str]:
    issues: List[str] = []
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            if (s < 0).any():
                issues.append(f"negative_{col}")
            if (s == 0).sum() > 0 and col != "volume":
                issues.append(f"zero_{col}")
    return sorted(list(set(issues)))


def detect_timestamp_order(df: pd.DataFrame) -> bool:
    if "date" not in df.columns:
        return True
    d = pd.to_datetime(df["date"]).astype("int64")
    return bool((np.diff(d) >= 0).all())


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate historical data for symbols")
    parser.add_argument("--symbols", type=str, required=True, help="Comma-separated symbols list")
    parser.add_argument("--out", type=str, default="reports/etl_validation.json", help="Output JSON report path")
    parser.add_argument("--z", type=float, default=8.0, help="Z-score threshold for outliers")
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    svc = AppService()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    report: dict = {"symbols": {}, "summary": {"critical": 0, "warnings": 0}}
    exit_code = 0

    for sym in symbols:
        try:
            df = svc.data_provider.get_daily_adjusted(sym)
            # Normalizar: asegurar columna date si index es fecha
            if "date" not in df.columns:
                idx = df.index
                if not isinstance(idx, pd.RangeIndex):
                    df = df.reset_index().rename(columns={df.columns[0]: "date"})
            gaps = detect_gaps(df)
            outliers = detect_outliers(df, z_thresh=float(args.z))
            bad_ticks = detect_bad_ticks(df)
            ts_ok = detect_timestamp_order(df)
            report["symbols"][sym] = {
                "rows": int(len(df)),
                "gaps": gaps,
                "outliers": outliers,
                "bad_ticks": bad_ticks,
                "timestamp_monotonic": bool(ts_ok),
            }
            critical = 0
            warnings = 0
            if not ts_ok:
                critical += 1
            if bad_ticks:
                critical += 1
            if gaps:
                warnings += len(gaps)
            if outliers:
                warnings += len(outliers)
            report["symbols"][sym]["critical"] = critical
            report["symbols"][sym]["warnings"] = warnings
            report["summary"]["critical"] += critical
            report["summary"]["warnings"] += warnings
        except Exception as exc:  # noqa: BLE001
            report["symbols"][sym] = {"error": str(exc)}
            report["summary"]["critical"] += 1
            exit_code = 1

    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    # Exit non-zero if any critical
    if int(report["summary"]["critical"]) > 0:
        exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())


