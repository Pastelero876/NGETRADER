from __future__ import annotations


def handle_task(task: dict) -> dict:
    """Orquesta tareas de análisis (stub IA externa opcional).

    task: {"type": str, "payload": dict}
    """
    t = (task or {}).get("type")
    p = (task or {}).get("payload", {})
    if t == "news_ingest":
        text = str(p.get("text", ""))
        tickers = [tk for tk in ["BTCUSDT", "ETHUSDT", "AAPL", "MSFT"] if tk in text]
        return {
            "signals": [{"ticker": tk, "sentiment": 0.0} for tk in tickers],
            "timestamp": p.get("timestamp"),
        }
    return {"status": "ignored"}


