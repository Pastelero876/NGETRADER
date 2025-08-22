from __future__ import annotations

from typing import List, Dict, Optional

import requests

from nge_trader.config.settings import Settings


class FREDService:
    """Cliente simple para FRED (series económicas)."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.settings = Settings()
        self.api_key = api_key or self.settings.fred_api_key
        self.base_url = "https://api.stlouisfed.org/fred"

    def _check_key(self) -> None:
        if not self.api_key:
            raise ValueError("Falta FRED_API_KEY en .env")

    def fetch_series_observations(
        self,
        series_id: str,
        observation_start: Optional[str] = None,
        observation_end: Optional[str] = None,
        limit: int = 2000,
    ) -> List[Dict]:
        self._check_key()
        url = f"{self.base_url}/series/observations"
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "limit": int(limit),
        }
        if observation_start:
            params["observation_start"] = observation_start
        if observation_end:
            params["observation_end"] = observation_end
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        obs = data.get("observations", [])
        out: List[Dict] = []
        for o in obs:
            out.append({"date": o.get("date"), "value": float(o.get("value") or 0.0)})
        return out


