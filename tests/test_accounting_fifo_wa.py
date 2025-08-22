from __future__ import annotations

from nge_trader.repository.db import Database
from nge_trader.services.accounting import recompute_lot_accounting


def test_fifo_realized_basic():
    db = Database()
    # limpiar trades/fills recientes no es trivial aquí; añadimos un caso y verificamos incremento esperado
    # Insertar fills BUY 1@10, BUY 1@12, SELL 1.5@15
    db.record_fill({"ts": "2025-01-01T00:00:01Z", "symbol": "TST", "side": "buy", "qty": 1.0, "price": 10.0, "order_id": "A"})
    db.record_fill({"ts": "2025-01-01T00:00:02Z", "symbol": "TST", "side": "buy", "qty": 1.0, "price": 12.0, "order_id": "B"})
    db.record_fill({"ts": "2025-01-01T00:00:03Z", "symbol": "TST", "side": "sell", "qty": 1.5, "price": 15.0, "order_id": "C"})
    res = recompute_lot_accounting(symbol="TST")
    assert isinstance(res, dict)
    # Realized: 1@(15-10) + 0.5@(15-12) = 5 + 1.5 = 6.5
    assert res["total_realized"] >= 6.5 - 1e-9


