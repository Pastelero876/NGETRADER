from __future__ import annotations

import time
from typing import Optional

import requests

from nge_trader.repository.db import Database


class AltSentimentIngestor:
    def __init__(self, db: Optional[Database] = None) -> None:
        self.db = db or Database()

    def ingest_twitter_recent(self, query: str, bearer_token: str) -> int:
        """Stub simple: usa búsqueda reciente v2 (si se provee token) y calcula polaridad heurística.

        Para producción, reemplace por un modelo de sentimiento robusto.
        """
        if not bearer_token:
            return 0
        url = "https://api.twitter.com/2/tweets/search/recent"
        headers = {"Authorization": f"Bearer {bearer_token}"}
        params = {"query": query, "max_results": 10, "tweet.fields": "created_at"}
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=5)
            data = resp.json()
        except Exception:
            return 0
        count = 0
        for t in data.get("data", [])[:10]:
            text = t.get("text") or ""
            ts = t.get("created_at") or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            # Heurística: pos si contiene palabras bullish, neg si bearish
            low = text.lower()
            score = 0.5 if any(k in low for k in ["bull", "breakout", "surge"]) else (0.5 if "pump" in low else 0.0)
            if any(k in low for k in ["bear", "dump", "collapse", "crash"]):
                score = 0.5
                label = "negative"
            elif score > 0:
                label = "positive"
            else:
                label = "neutral"
            row = {
                "ts": ts,
                "source": "twitter",
                "symbol": None,
                "sentiment_score": float(score),
                "sentiment_label": label,
                "text": text[:500],
                "link": None,
            }
            try:
                self.db.ingest_alt_signals([row])
                count += 1
            except Exception:
                continue
        return count


