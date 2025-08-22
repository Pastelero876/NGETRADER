import json

from nge_trader.repository.db import Database


def test_metrics_storage_roundtrip():
    db = Database()
    db.put_metric("hit_rate", 0.55)
    db.put_metric("sharpe_live", 1.23)
    series = db.recent_metric_series("hit_rate", 5)
    assert series
    assert series[-1][1] == 0.55


def test_recent_metric_values_from_logs():
    db = Database()
    db.append_log_json("INFO", {"metric": "slippage_bps", "value": 12.3}, "2024-01-01T00:00:00Z")
    vals = db.recent_metric_values("slippage_bps", 10)
    assert any(abs(v) >= 12.3 for _, v in vals)


def test_trade_realized_includes_fees_on_sell():
    db = Database()
    fill = {
        "ts": "2020-01-01T00:00:00Z",
        "symbol": "AAPL",
        "side": "sell",
        "qty": 1.0,
        "price": 100.0,
        "order_id": "x1",
        "fees": 0.5,
    }
    db.record_fill(fill)
    tr = db.recent_trades(1)[0]
    assert abs(float(tr.get("realized") or 0.0) - 99.5) < 1e-9


def test_rate_limiter_token_bucket():
    from nge_trader.services.rate_limiter import GlobalRateLimiter
    rl = GlobalRateLimiter.get()
    rl.configure("orders", capacity=2, refill_per_sec=0.0)
    assert rl.acquire("orders") is True
    assert rl.acquire("orders") is True
    assert rl.acquire("orders") is False

def test_metrics_exporter_string():
	from nge_trader.services.metrics import export_metrics_text
	txt = export_metrics_text()
	assert isinstance(txt, str)