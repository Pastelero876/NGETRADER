from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from huggingface_hub import InferenceClient

from nge_trader.config.settings import Settings


class HFTimeseriesForecaster:
    """Wrapper opcional para modelos HF de predicción de series temporales vía Inference API.

    Este es un stub genérico que envía una ventana de precios y recibe una predicción de retorno/valor.
    Requiere `HUGGINGFACE_API_TOKEN`.
    """

    def __init__(self, model: str = "facebook/prophet") -> None:
        self.settings = Settings()
        if not self.settings.huggingface_api_token:
            raise RuntimeError("Se requiere HUGGINGFACE_API_TOKEN para usar el forecaster HF")
        self.client = InferenceClient(token=self.settings.huggingface_api_token)
        self.model = model

    def predict_next_return(self, df: pd.DataFrame, lookback: int = 60) -> float:
        close = df["close"].astype(float).values
        if len(close) < lookback + 1:
            return 0.0
        window = close[-lookback:]
        payload = {"inputs": {"series": window.tolist()}}
        try:
            out = self.client.post(json=payload, model=self.model, task="time-series-forecasting")
            # se espera un dict con "prediction"
            if isinstance(out, dict) and "prediction" in out:
                return float(out["prediction"])  # retorno o delta según modelo
        except Exception:
            pass
        # fallback: retorno simple
        return float(np.log(close[-1] / close[-2]))


