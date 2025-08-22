from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import json
import os
from pathlib import Path

import pandas as pd

from nge_trader.config.settings import Settings
from nge_trader.services.app_service import AppService
from nge_trader.repository.db import Database


@dataclass
class DatasetSpec:
    symbols: List[str]
    timeframes: List[str]  # e.g., ["15m", "1h", "4h"]
    years: int = 3
    dataset_root: str = "datasets/ohlcv"
    version: Optional[str] = None  # if None, timestamped


def _ensure_timestamps_utc(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
        out = out.dropna(subset=["timestamp"])  # remove invalid
        out = out.sort_values("timestamp").reset_index(drop=True)
    elif "date" in out.columns:
        out["timestamp"] = pd.to_datetime(out["date"], utc=True, errors="coerce")
        out = out.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    else:
        out["timestamp"] = pd.to_datetime(pd.Timestamp.utcnow(), utc=True)
    return out


def _clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    out = _ensure_timestamps_utc(df)
    cols = [c for c in ["open", "high", "low", "close", "volume"] if c in out.columns]
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=[c for c in cols if c != "volume"]).reset_index(drop=True)
    if "timestamp" in out.columns:
        out = out.drop_duplicates(subset=["timestamp"])  # dedupe same bar
    return out


def _expected_freq_str(tf: str) -> str:
    tf = tf.lower().strip()
    if tf.endswith("m"):
        return f"{int(tf[:-1])}T"  # pandas minute alias
    if tf.endswith("h"):
        return f"{int(tf[:-1])}H"
    if tf in ("d", "1d", "day"):
        return "1D"
    return "1D"


def _detect_gaps(df: pd.DataFrame, tf: str) -> Dict[str, Any]:
    if df.empty:
        return {"count": 0, "max_gap": 0}
    if "timestamp" not in df.columns:
        return {"count": 0, "max_gap": 0}
    s = df["timestamp"].sort_values().reset_index(drop=True)
    freq = _expected_freq_str(tf)
    try:
        expected = pd.date_range(start=s.iloc[0], end=s.iloc[-1], freq=freq, inclusive="both")
        present = pd.DatetimeIndex(s.values)
        missing = expected.difference(present)
        # rough max gap in number of expected steps between consecutive present timestamps
        diffs = present.to_series().diff().dropna().dt.total_seconds().values.tolist()
        step = pd.to_timedelta(freq).total_seconds() if hasattr(pd, "to_timedelta") else 0
        max_gap = max([d / step for d in diffs]) if diffs and step else 0
        return {"count": int(len(missing)), "max_gap": float(max_gap)}
    except Exception:
        return {"count": 0, "max_gap": 0}


def _resample_to_timeframe(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    if df.empty:
        return df
    if "timestamp" not in df.columns:
        return df
    out = df.copy()
    out = out.set_index("timestamp")
    rule = _expected_freq_str(tf)
    # Standard OHLCV resample
    ohlc = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    cols = [c for c in ohlc.keys() if c in out.columns]
    agg = {c: ohlc[c] for c in cols}
    res = out.resample(rule).agg(agg).dropna().reset_index()
    return res


def _estimate_spread_bps(df: pd.DataFrame) -> Optional[float]:
    if df.empty:
        return None
    if "bid" in df.columns and "ask" in df.columns:
        mid = (pd.to_numeric(df["bid"]) + pd.to_numeric(df["ask"])) / 2.0
        spread = (pd.to_numeric(df["ask"]) - pd.to_numeric(df["bid"])) / mid.replace(0, pd.NA)
        bps = (spread * 10000.0).dropna()
        return float(bps.median()) if not bps.empty else None
    # Proxy from OHLC as fallback (overestimates): intrabar range as proxy of liquidity/vol
    if set(["high", "low", "close"]).issubset(df.columns):
        close = pd.to_numeric(df["close"])\
            .replace([pd.NA, pd.NaT], 0).astype(float)
        rng = (pd.to_numeric(df["high"]) - pd.to_numeric(df["low"])) / close.replace(0, pd.NA)
        bps = (rng * 10000.0 / 4.0).dropna()  # heuristic
        return float(bps.median()) if not bps.empty else None
    return None


def _estimate_fees_bps(symbol: str) -> Dict[str, float]:
    try:
        db = Database()
        row = db.get_fee_schedule_any("DEFAULT", symbol)
        if row:
            return {
                "maker_bps": float(row.get("maker_bps") or 5.0),
                "taker_bps": float(row.get("taker_bps") or 10.0),
            }
    except Exception:
        pass
    return {"maker_bps": 5.0, "taker_bps": 10.0}


def build_dataset(spec: DatasetSpec) -> Dict[str, Any]:
    s = Settings()
    app = AppService()
    version = spec.version or pd.Timestamp.utcnow().strftime("v%Y%m%d%H%M%S")
    root = Path(spec.dataset_root) / version
    manifest: Dict[str, Any] = {
        "version": version,
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "symbols": list(spec.symbols),
        "timeframes": list(spec.timeframes),
        "years": int(spec.years),
        "provider": s.data_provider,
        "issues": {},
        "costs": {},
        "partitions": [],
    }
    root.mkdir(parents=True, exist_ok=True)

    for symbol in spec.symbols:
        try:
            # Fetch base (daily as fallback)
            base = app.data_provider.get_daily_adjusted(symbol)
        except Exception:
            base = pd.DataFrame()
        base = _clean_ohlcv(base)
        if not base.empty:
            # Limit history to N years
            cutoff = pd.Timestamp.utcnow().tz_localize("UTC").normalize() - pd.Timedelta(days=365 * max(spec.years, 1))
            base = base[base["timestamp"] >= cutoff]
        # Costs estimation (proxy)
        spread_bps = _estimate_spread_bps(base)
        fees = _estimate_fees_bps(symbol)
        manifest["costs"][symbol] = {
            "spread_bps_median": spread_bps,
            "maker_bps": fees["maker_bps"],
            "taker_bps": fees["taker_bps"],
        }
        for tf in spec.timeframes:
            df_tf = _resample_to_timeframe(base, tf) if not base.empty else base
            issues = _detect_gaps(df_tf, tf)
            manifest["issues"].setdefault(symbol, {})[tf] = issues
            # Write partitioned parquet
            part_dir = root / tf / f"symbol={symbol.upper()}"
            part_dir.mkdir(parents=True, exist_ok=True)
            if not df_tf.empty:
                # Partition by date=YYYY-MM-DD
                df_tf["date"] = df_tf["timestamp"].dt.date.astype(str)
                for d, grp in df_tf.groupby("date"):
                    out_path = part_dir / f"date={d}" / "data.parquet"
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    grp.drop(columns=[c for c in ["date"] if c in grp.columns]).to_parquet(str(out_path), index=False)
                    manifest["partitions"].append({"symbol": symbol, "tf": tf, "date": d, "path": str(out_path)})
            else:
                # Create empty placeholder
                (part_dir / "_EMPTY").touch()

    # Save manifest
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


