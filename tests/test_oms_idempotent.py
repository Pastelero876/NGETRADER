from __future__ import annotations

from nge_trader.repository.db import Database
from nge_trader.services.oms import cancel_order_idempotent, replace_order_limit_price_idempotent


class _DummyBroker:
    def cancel_order(self, order_id: str):  # noqa: ANN001
        return {"status": "canceled", "order_id": order_id}

    def replace_order(self, order_id: str, new_price: float):  # noqa: ANN001
        return {"status": "replaced", "order_id": order_id, "price": float(new_price)}


def test_cancel_order_idempotent_inserts_outbox_and_updates():
    db = Database()
    before = db.count_outbox_by_status().get("sent", 0)
    res = cancel_order_idempotent(_DummyBroker(), order_id="OID123", symbol="TEST")
    assert isinstance(res, dict)
    after = db.count_outbox_by_status().get("sent", 0)
    assert after >= before + 1


def test_replace_order_idempotent_inserts_outbox_and_updates():
    db = Database()
    before = db.count_outbox_by_status().get("sent", 0)
    res = replace_order_limit_price_idempotent(_DummyBroker(), order_id="OID456", new_price=123.45, symbol="TEST")
    assert isinstance(res, dict)
    after = db.count_outbox_by_status().get("sent", 0)
    assert after >= before + 1


