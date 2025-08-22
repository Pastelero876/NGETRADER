from __future__ import annotations

from typing import Dict, List, Optional

import requests

from nge_trader.config.settings import Settings


class TwitterService:
    """Cliente simple de Twitter API v2 (búsqueda reciente)."""

    def __init__(self, bearer_token: Optional[str] = None) -> None:
        self.settings = Settings()
        self.bearer = bearer_token or self.settings.twitter_bearer_token
        if not self.bearer:
            raise ValueError("Falta TWITTER_BEARER_TOKEN en .env")
        self.base_url = "https://api.twitter.com/2"
        self.headers = {"Authorization": f"Bearer {self.bearer}"}

    def search_recent(self, query: str, max_results: int = 20) -> List[Dict]:
        url = f"{self.base_url}/tweets/search/recent"
        params = {
            "query": query,
            "max_results": max(10, min(int(max_results), 100)),
            "tweet.fields": "created_at,text,lang",
        }
        resp = requests.get(url, headers=self.headers, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return list(data.get("data", []))


