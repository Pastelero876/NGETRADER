from __future__ import annotations

from typing import Any, Dict, List

import requests
import pandas as pd


class TiingoDataProvider:
    """Proveedor de datos de Tiingo (requiere API key en TIINGO_API_KEY).

    Cubre acciones, cripto y FX. Aquí implementamos un fetch diario básico.
    """

    def __init__(self, api_key: str | None) -> None:
        if not api_key:
            raise ValueError("Falta TIINGO_API_KEY")
        self.api_key = api_key
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})

    def get_daily_adjusted(self, symbol: str, limit: int = 5000) -> pd.DataFrame:
        url = f"https://api.tiingo.com/tiingo/daily/{symbol}/prices"
        params = {"token": self.api_key, "format": "json"}
        resp = self._session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data: List[Dict[str, Any]] = resp.json()
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        # Normalizar nombres
        df = df.rename(columns={"adjClose": "close", "adjOpen": "open", "adjHigh": "high", "adjLow": "low", "adjVolume": "volume"})
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])  # type: ignore[assignment]
        cols = [c for c in ["date","open","high","low","close","volume"] if c in df.columns]
        return df[cols]


