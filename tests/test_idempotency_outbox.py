from __future__ import annotations

import json

from nge_trader.repository.db import Database
from nge_trader.services import metrics as metrics_mod


def test_idempotency_outbox_and_duplicates():
    db = Database()
    payload = json.dumps({"action": "test", "ts": "2025-01-01T00:00:00Z"})
    idem_key = "test_idem_key_123"
    out_id = db.put_order_outbox(payload, idempotency_key=idem_key, correlation_id="corr_test")
    assert isinstance(out_id, int) and out_id > 0

    # Primera vez: registra seen
    db.record_idempotency_seen(idem_key, status="seen")
    assert db.idempotency_exists(idem_key) is True

    # Segunda vez: cuenta duplicado en métricas
    before = float(metrics_mod._METRICS_STORE.get("idempotency_duplicates_total", 0.0))
    db.record_idempotency_seen(idem_key, status="seen")
    after = float(metrics_mod._METRICS_STORE.get("idempotency_duplicates_total", 0.0))
    assert after >= before + 1.0


def test_order_outbox_and_idempotency():
    db = Database()
    payload = json.dumps({"symbol": "TEST", "side": "buy", "quantity": 1.0})
    key = "TEST_buy_dummy"
    corr = "corr_test"
    outbox_id = db.put_order_outbox(payload, key, corr)
    assert isinstance(outbox_id, int) and outbox_id > 0
    db.mark_order_outbox(outbox_id, status="sent", broker_order_id="B123")
    row = db.get_outbox_row(outbox_id)
    assert row is not None and row["status"] == "sent" and row["broker_order_id"] == "B123"
    db.record_idempotency_seen(key, status="sent")
    assert db.idempotency_exists(key) is True
    counts = db.count_outbox_by_status()
    assert isinstance(counts, dict) and counts.get("sent", 0) >= 1


