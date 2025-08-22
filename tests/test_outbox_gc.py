from __future__ import annotations

import json


def test_outbox_gc_deletes_old_rows():
    from nge_trader.repository.db import Database
    import datetime as dt

    db = Database()
    # Insertar 2 filas en outbox con ts antiguo
    o1 = db.put_order_outbox(json.dumps({"x": 1}), idempotency_key="K1", correlation_id="C1")
    o2 = db.put_order_outbox(json.dumps({"x": 2}), idempotency_key="K2", correlation_id="C2")
    db.mark_order_outbox(o1, status="error")
    db.mark_order_outbox(o2, status="sent")
    # Retroceder timestamps 10 días
    old_ts = (dt.datetime.now(dt.UTC) - dt.timedelta(days=10)).isoformat()
    db.update_outbox_ts(o1, old_ts)
    db.update_outbox_ts(o2, old_ts)
    # GC con max_age_days=7 debe borrar ambas
    deleted = db.gc_order_outbox(max_age_days=7, statuses=("sent", "error"))
    assert deleted >= 2


