from __future__ import annotations

import pandas as pd

from nge_trader.repository.db import Database


def enrich_features(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    out = pd.DataFrame()
    close = df["close"].astype(float).reset_index(drop=True)
    out["log_ret"] = (close.apply(lambda x: max(float(x), 1e-9)).apply(pd.np.log).diff()).fillna(0.0)  # type: ignore[attr-defined]
    out["ret"] = close.pct_change().fillna(0.0)
    out["roll_mean"] = close.rolling(5).mean().fillna(method="bfill")
    out["roll_std"] = close.rolling(5).std(ddof=0).fillna(method="bfill").replace(0.0, 1e-9)
    # gap (open vs prev close) si existe open
    if "open" in df.columns:
        open_ = df["open"].astype(float).reset_index(drop=True)
        prev_close = close.shift(1)
        out["gap_pct"] = (open_ - prev_close) / prev_close.replace(0.0, 1e-9)
    else:
        out["gap_pct"] = 0.0
    # volumen z-score
    if "volume" in df.columns:
        vol = df["volume"].astype(float).reset_index(drop=True)
        vol_ma = vol.rolling(window).mean()
        vol_std = vol.rolling(window).std(ddof=0).replace(0.0, 1e-9)
        out["vol_z"] = (vol - vol_ma) / vol_std
    else:
        out["vol_z"] = 0.0
    return out.fillna(0.0)


def save_features_to_store(symbol: str, features: pd.DataFrame) -> None:
    if features.empty:
        return
    db = Database()
    rows = []
    for idx, row in features.iterrows():
        ts = str(idx)
        for col in features.columns:
            rows.append({
                "ts": ts,
                "symbol": symbol,
                "feature": col,
                "value": float(row[col] if pd.notna(row[col]) else 0.0),
            })
    db.save_features(rows)


