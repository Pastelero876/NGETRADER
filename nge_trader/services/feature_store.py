from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import json
import pandas as pd

from nge_trader.config.settings import Settings


@dataclass
class FeatureSpec:
    dataset_root: str
    version: str
    out_root: Optional[str] = None  # default: <dataset_root>/<version>/features
    horizons: List[int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.horizons is None:
            self.horizons = [5, 20]


def _safe_pct_change(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    return s.pct_change().fillna(0.0)


def compute_regime_tags(df: pd.DataFrame, short: int = 10, long: int = 30) -> pd.Series:
    close = pd.to_numeric(df["close"], errors="coerce").astype(float)
    mom = (close / close.shift(long) - 1.0).fillna(0.0)
    vol = _safe_pct_change(close).rolling(long).std(ddof=0).fillna(0.0)
    # 0: lateral, 1: tendencia positiva, -1: negativa
    regime = pd.Series(0, index=df.index)
    regime = regime.mask(mom > vol, 1)
    regime = regime.mask(mom < -vol, -1)
    return regime.fillna(0).astype(int)


def compute_features(df: pd.DataFrame, horizons: List[int]) -> pd.DataFrame:
    out = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    close = pd.to_numeric(out["close"], errors="coerce").astype(float)
    ret1 = _safe_pct_change(close)
    out["ret_1"] = ret1
    for h in horizons:
        vol_h = ret1.rolling(h).std(ddof=0)
        mean_h = ret1.rolling(h).mean()
        out[f"vol_{h}"] = vol_h
        out[f"ret_mean_{h}"] = mean_h
        out[f"zret_{h}"] = (ret1 - mean_h) / (vol_h.replace(0, pd.NA))
    out["regime"] = compute_regime_tags(out)
    return out


def fit_scaler(df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, Dict[str, float]]:
    scaler: Dict[str, Dict[str, float]] = {}
    for c in feature_cols:
        s = pd.to_numeric(df[c], errors="coerce").astype(float)
        mean = float(s.mean()) if len(s) else 0.0
        std = float(s.std(ddof=0)) if len(s) else 1.0
        if std == 0:
            std = 1.0
        scaler[c] = {"mean": mean, "std": std}
    return scaler


def apply_scaler(df: pd.DataFrame, scaler: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    out = df.copy()
    for c, st in scaler.items():
        if c in out.columns:
            out[c] = (pd.to_numeric(out[c], errors="coerce").astype(float) - float(st["mean"])) / float(st["std"])
    return out


def compute_reward_labels(df: pd.DataFrame, horizons: List[int]) -> pd.DataFrame:
    s = Settings()
    out = df.copy()
    close = pd.to_numeric(out["close"], errors="coerce").astype(float)
    ret1 = _safe_pct_change(close)
    # Cost proxies
    slip_bps = float(getattr(s, "lambda_slippage", 0.1) or 0.1)  # weight, not actual slippage
    fee_bps = float(getattr(s, "lambda_fees", 1.0) or 1.0)
    # approximate drawdown penalty via rolling min of cumulative returns
    cum = (1.0 + ret1).cumprod()
    roll_min = cum.rolling(50).min().fillna(cum)
    dd = (cum / roll_min - 1.0)
    lam_dd = float(getattr(s, "lambda_drawdown", 0.2) or 0.2)
    # Normalize by ATR-like proxy (rolling volatility)
    atrp = ret1.rolling(14).std(ddof=0).replace(0, pd.NA).fillna(method="bfill").fillna(method="ffill").replace(0, 1e-6)
    reward = ret1 - (slip_bps / 10000.0) - (fee_bps / 10000.0) - lam_dd * dd
    reward = reward / atrp
    out["reward"] = reward.fillna(0.0)
    # Include explicit "no-trade" action value by clipping small rewards to zero
    out["reward_no_trade"] = out["reward"].clip(lower=0.0)
    return out


def build_feature_store(spec: FeatureSpec) -> Dict[str, Any]:
    root = Path(spec.dataset_root) / spec.version
    manifest_path = root / "manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    out_root = Path(spec.out_root) if spec.out_root else (root / "features")
    out_root.mkdir(parents=True, exist_ok=True)
    scalers_root = out_root / "_scalers"
    scalers_root.mkdir(parents=True, exist_ok=True)

    parts = data.get("partitions", [])
    processed = 0
    feature_cols: List[str] = []
    for p in parts:
        symbol = p.get("symbol")
        tf = p.get("tf")
        path = p.get("path")
        if not (symbol and tf and path):
            continue
        df = pd.read_parquet(path) if Path(path).exists() else pd.DataFrame()
        if df.empty:
            continue
        feats = compute_features(df, horizons=spec.horizons)
        feats = compute_reward_labels(feats, horizons=spec.horizons)
        # Update feature columns once
        if not feature_cols:
            feature_cols = [c for c in feats.columns if c not in ("timestamp", "open", "high", "low", "close", "volume")]
        # Load or fit scaler per (symbol, tf)
        scaler_path = scalers_root / f"symbol={symbol.upper()}__tf={tf}.json"
        if scaler_path.exists():
            scaler = json.loads(scaler_path.read_text(encoding="utf-8"))
        else:
            scaler = fit_scaler(feats, feature_cols)
            scaler_path.write_text(json.dumps(scaler, ensure_ascii=False, indent=2), encoding="utf-8")
        feats_s = apply_scaler(feats, scaler)
        # Save partitioned features
        out_dir = out_root / tf / f"symbol={symbol.upper()}" / f"date={Path(path).parent.name.split('=')[-1]}"
        out_dir.mkdir(parents=True, exist_ok=True)
        feats_s.to_parquet(str(out_dir / "features.parquet"), index=False)
        processed += 1

    # Save feature manifest
    fmanifest = {
        "dataset_version": spec.version,
        "root": str(out_root),
        "scalers": str(scalers_root),
        "processed_partitions": int(processed),
        "horizons": list(spec.horizons),
    }
    (out_root / "manifest.json").write_text(json.dumps(fmanifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return fmanifest


