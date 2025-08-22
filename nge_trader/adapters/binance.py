from __future__ import annotations

from typing import Any, Dict, List

import requests
import pandas as pd


class BinanceDataProvider:
    """Proveedor simple de datos diarios (spot) desde Binance.

    Usa klines 1d públicos para símbolos como BTCUSDT, ETHUSDT, etc.
    """

    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = base_url or "https://api.binance.com"
        self._session = requests.Session()

    def get_daily_adjusted(self, symbol: str, limit: int = 1000) -> pd.DataFrame:
        url = f"{self.base_url}/api/v3/klines"
        params = {"symbol": symbol.upper(), "interval": "1d", "limit": int(limit)}
        resp = self._session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data: List[List[Any]] = resp.json()
        if not data:
            return pd.DataFrame()
        rows: List[Dict[str, Any]] = []
        for k in data:
            # [ openTime, open, high, low, close, volume, closeTime, ... ]
            rows.append(
                {
                    "date": pd.to_datetime(k[0], unit="ms"),
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                }
            )
        df = pd.DataFrame(rows)
        return df


