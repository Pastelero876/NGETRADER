from __future__ import annotations

import pandas as pd


def apply_one_day_shock(df: pd.DataFrame, shock_pct: float, date: pd.Timestamp | None = None) -> pd.DataFrame:
    if df.empty or shock_pct == 0.0:
        return df
    out = df.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"])  # type: ignore[assignment]
        out = out.sort_values("date").reset_index(drop=True)
        if date is None:
            idx = len(out) - 1
        else:
            try:
                idx = int(out.index[out["date"] == date][0])
            except Exception:
                idx = len(out) - 1
    else:
        idx = len(out) - 1
    cols = [c for c in ["open", "high", "low", "close"] if c in out.columns]
    for c in cols:
        out.loc[idx, c] = float(out.loc[idx, c]) * (1.0 + shock_pct)
    return out


