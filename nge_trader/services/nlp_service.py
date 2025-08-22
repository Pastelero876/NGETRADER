from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict

import pandas as pd
import requests
import hashlib
import json

from nge_trader.config.settings import Settings
from nge_trader.repository.db import Database


class NLPService:
    def __init__(self) -> None:
        self.settings = Settings()
        self.base = "https://api-inference.huggingface.co/models"
        self.headers = {"Authorization": f"Bearer {self.settings.huggingface_api_token}"} if self.settings.huggingface_api_token else None

    def _post(self, model: str, payload: Dict) -> Dict:
        if not self.headers:
            raise ValueError("Falta HUGGINGFACE_API_TOKEN. Ve a Configuración y añade tu token de Hugging Face.")
        url = f"{self.base}/{model}"
        resp = requests.post(url, headers=self.headers, json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()

    def analyze_sentiment(self, texts: List[str]) -> List[Dict]:
        if not texts:
            return []
        model = self.settings.huggingface_sentiment_model
        key_hash = hashlib.sha256((model + "::" + "||".join(texts)).encode("utf-8")).hexdigest()
        cached = Database().get_nlp_cache("sentiment", model, key_hash)
        if cached:
            try:
                return json.loads(cached)
            except Exception:
                pass
        out = self._post(model, {"inputs": texts})
        # Normalizar
        results = []
        for item in out:
            # item es lista de labels
            best = max(item, key=lambda x: x.get("score", 0)) if isinstance(item, list) else item
            results.append({"label": best.get("label"), "score": best.get("score")})
        try:
            Database().put_nlp_cache("sentiment", model, key_hash, json.dumps(results))
        except Exception:
            pass
        return results

    def summarize(self, text: str, max_tokens: int = 128) -> str:
        model = self.settings.huggingface_summarization_model
        key_hash = hashlib.sha256((model + "::" + text + f"::{max_tokens}").encode("utf-8")).hexdigest()
        cached = Database().get_nlp_cache("summary", model, key_hash)
        if cached is not None:
            return cached
        out = self._post(model, {"inputs": text, "parameters": {"max_new_tokens": max_tokens}})
        if isinstance(out, list) and out:
            result = out[0].get("summary_text") or out[0].get("generated_text") or ""
            try:
                Database().put_nlp_cache("summary", model, key_hash, result)
            except Exception:
                pass
            return result
        return ""

    def embed(self, texts: List[str]) -> pd.DataFrame:
        if not texts:
            return pd.DataFrame()
        model = self.settings.huggingface_embedding_model
        key_hash = hashlib.sha256((model + "::" + "||".join(texts)).encode("utf-8")).hexdigest()
        cached = Database().get_nlp_cache("embed", model, key_hash)
        if cached:
            try:
                arr = json.loads(cached)
                return pd.DataFrame(arr)
            except Exception:
                pass
        out = self._post(model, {"inputs": texts})
        # El API suele devolver lista de lista de vectores
        vecs = []
        for item in out:
            # tomar el pool (primer vector)
            v = item[0] if isinstance(item, list) and item else []
            vecs.append(pd.Series(v).astype(float))
        df = pd.DataFrame(vecs)
        df.columns = [f"dim_{i}" for i in range(df.shape[1])] if not df.empty else []
        try:
            Database().put_nlp_cache("embed", model, key_hash, json.dumps(df.values.tolist()))
        except Exception:
            pass
        return df


