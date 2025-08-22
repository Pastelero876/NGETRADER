from __future__ import annotations

import requests
import pandas as pd
from nge_trader.services.rate_limiter import GlobalRateLimiter


class AlphaVantageDataProvider:
    """Proveedor de datos de mercado usando Alpha Vantage."""

    def __init__(self, api_key: str | None) -> None:
        self.api_key = api_key
        # Límite típico free: 5 req/min
        # Configurar bucket global (clave 'alpha_vantage')
        GlobalRateLimiter.get().configure("alpha_vantage", capacity=5, refill_per_sec=5/60.0)

    def get_daily_adjusted(self, symbol: str) -> pd.DataFrame:
        """Descarga series diarias ajustadas y devuelve un DataFrame ordenado por fecha.

        Mensajes de error descriptivos en español.
        """

        if not self.api_key:
            raise ValueError("Falta la API key de Alpha Vantage (ALPHA_VANTAGE_API_KEY)")

        url = "https://www.alphavantage.co/query"
        params = {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": symbol,
            "outputsize": "full",
            "apikey": self.api_key,
        }
        GlobalRateLimiter.get().acquire("alpha_vantage")
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        key = "Time Series (Daily)"
        if key not in data:
            detail = data.get("Note") or data.get("Error Message") or "sin detalles"
            raise ValueError(f"Respuesta inesperada de Alpha Vantage: {detail}")

        time_series = data[key]
        records: list[dict] = []
        for date_str, values in time_series.items():
            records.append(
                {
                    "date": pd.to_datetime(date_str),
                    "open": float(values["1. open"]),
                    "high": float(values["2. high"]),
                    "low": float(values["3. low"]),
                    "close": float(values["4. close"]),
                    "adjusted_close": float(values["5. adjusted close"]),
                    "volume": int(values["6. volume"]),
                }
            )

        df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
        return df

    def get_intraday(self, symbol: str, interval: str = "5min", outputsize: str = "compact") -> pd.DataFrame:
        """Descarga series intradía (1min/5min/15min/30min/60min)."""
        if not self.api_key:
            raise ValueError("Falta la API key de Alpha Vantage (ALPHA_VANTAGE_API_KEY)")
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": symbol,
            "interval": interval,
            "outputsize": outputsize,
            "apikey": self.api_key,
        }
        GlobalRateLimiter.get().acquire("alpha_vantage")
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        key = f"Time Series ({interval})"
        if key not in data:
            detail = data.get("Note") or data.get("Error Message") or "sin detalles"
            raise ValueError(f"Respuesta inesperada de Alpha Vantage (intraday): {detail}")
        ts = data[key]
        rows: list[dict] = []
        for dt, v in ts.items():
            rows.append({
                "date": pd.to_datetime(dt),
                "open": float(v["1. open"]),
                "high": float(v["2. high"]),
                "low": float(v["3. low"]),
                "close": float(v["4. close"]),
                "volume": float(v.get("5. volume", 0)),
            })
        df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
        return df


