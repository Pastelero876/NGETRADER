from __future__ import annotations

from nge_trader.repository.db import Database


def test_set_and_get_fee_schedule():
    db = Database()
    db.set_fee_schedule("BINANCE", "BTCUSDT", maker_bps=1.0, taker_bps=5.0, tier="vip0")
    row = db.get_fee_schedule("BINANCE", "BTCUSDT", tier="vip0")
    assert row is not None and float(row["maker_bps"]) == 1.0 and float(row["taker_bps"]) == 5.0


def test_record_fill_applies_fee_schedule_when_missing():
    db = Database()
    db.set_fee_schedule("BINANCE", "BTCUSDT", maker_bps=1.0, taker_bps=5.0, tier="vip0")
    # fees ausentes -> calcular por taker_bps
    db.record_fill({
        "ts": "2025-01-01T00:00:00Z",
        "exchange": "BINANCE",
        "symbol": "BTCUSDT",
        "side": "buy",
        "qty": 1.0,
        "price": 100.0,
        "order_id": "XX1",
    })
    f = db.recent_fills(1)[0]
    assert abs(float(f.get("fees") or 0.0) - 0.05) < 1e-9


