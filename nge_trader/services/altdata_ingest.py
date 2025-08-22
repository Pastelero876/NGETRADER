from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict

import feedparser
import pandas as pd

from nge_trader.services.nlp_service import NLPService


@dataclass
class Article:
    title: str
    summary: str
    link: str
    published: str


class AltDataIngest:
    def __init__(self) -> None:
        self.nlp = NLPService()

    def fetch_rss(self, url: str, limit: int = 20) -> List[Article]:
        feed = feedparser.parse(url)
        arts: List[Article] = []
        for e in feed.entries[:limit]:
            arts.append(
                Article(
                    title=e.get("title", ""),
                    summary=e.get("summary", ""),
                    link=e.get("link", ""),
                    published=str(e.get("published", "")),
                )
            )
        return arts

    def analyze_articles(self, articles: List[Article]) -> pd.DataFrame:
        if not articles:
            return pd.DataFrame()
        texts = [a.title + ". " + a.summary for a in articles]
        sent = self.nlp.analyze_sentiment(texts)
        sums = [self.nlp.summarize(t) for t in texts]
        emb = self.nlp.embed(texts)
        df = pd.DataFrame(
            {
                "title": [a.title for a in articles],
                "summary": [a.summary for a in articles],
                "link": [a.link for a in articles],
                "published": [a.published for a in articles],
                "sent_label": [s.get("label") for s in sent],
                "sent_score": [s.get("score") for s in sent],
                "gen_summary": sums,
            }
        )
        df = pd.concat([df, emb.reset_index(drop=True)], axis=1)
        return df


