from __future__ import annotations

from pathlib import Path

import pandas as pd

from nge_trader.config.settings import Settings


class DataCache:
    def __init__(self) -> None:
        self.settings = Settings()
        self.base = Path("data/cache")
        self.base.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.base / f"{key}.parquet"

    def load_df(self, key: str) -> pd.DataFrame | None:
        p = self._path(key)
        if not p.exists():
            return None
        age = pd.Timestamp.now(tz=pd.Timestamp.utcnow().tz) - pd.Timestamp(p.stat().st_mtime, unit="s", tz="UTC")
        if age.total_seconds() > self.settings.cache_ttl_seconds:
            return None
        try:
            return pd.read_parquet(p)
        except Exception:
            return None

    def save_df(self, key: str, df: pd.DataFrame) -> None:
        p = self._path(key)
        try:
            df.to_parquet(p, index=False)
        except Exception:
            # fallback CSV
            p = p.with_suffix(".csv")
            df.to_csv(p, index=False)


