from __future__ import annotations

from typing import List, Dict

import feedparser


SEC_FEED = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&owner=include&count=100&output=atom"


def fetch_recent_filings() -> List[Dict]:
    feed = feedparser.parse(SEC_FEED)
    out: List[Dict] = []
    for e in feed.entries[:50]:
        out.append(
            {
                "title": e.get("title", ""),
                "summary": e.get("summary", ""),
                "link": e.get("link", ""),
                "updated": e.get("updated", ""),
            }
        )
    return out


