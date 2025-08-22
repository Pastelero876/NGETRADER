from __future__ import annotations

import sys
from typing import Optional

from nge_trader.services.app_service import AppService
from nge_trader.repository.db import Database


def main(symbol: Optional[str] = None) -> None:
    svc = AppService()
    sym = symbol or "AAPL"
    # Backfill incremental: usa `backfill_for_symbol` (aplica dedup por order_id/price/qty y actualiza estados)
    try:
        res = svc.backfill_for_symbol(sym, limit=200)
    except Exception:
        res = {"orders": 0, "fills": 0}
    # GC outbox antiguo
    try:
        deleted = Database().gc_order_outbox(max_age_days=7, statuses=("sent", "error"))
        res["outbox_gc_deleted"] = int(deleted)
    except Exception:
        pass
    print({"status": "ok", "symbol": sym, **res})
    sys.exit(0)


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    main(arg)


