from __future__ import annotations

import json
import sys

from nge_trader.services.app_service import AppService

try:
    from prometheus_client import Counter  # type: ignore
except Exception:  # pragma: no cover
    Counter = None  # type: ignore

recon_mismatches_total = Counter("reconciliation_mismatches_total", "mismatches por tipo", ["type"]) if Counter else None  # type: ignore


def main() -> int:
    app = AppService()
    out = app.reconcile_state(resolve=False)
    # out puede ser dict con listas o contadores por tipo; normalizamos a contadores
    totals: dict[str, int] = {}
    if isinstance(out, dict):
        for k, v in out.items():
            if isinstance(v, list):
                totals[k] = len(v)
            else:
                try:
                    totals[k] = int(v)
                except Exception:
                    totals[k] = 0
    mismatches = sum(totals.values())
    # Métrica
    if recon_mismatches_total is not None:
        for t, n in totals.items():
            try:
                recon_mismatches_total.labels(t).inc(n)  # type: ignore[attr-defined]
            except Exception:
                pass
    print(json.dumps({"mismatches": totals, "total": mismatches}))
    return 0 if mismatches == 0 else 2


if __name__ == "__main__":
    sys.exit(main())

from __future__ import annotations

import sys
from nge_trader.services.app_service import AppService
from nge_trader.services.notifier import Notifier
from nge_trader.repository.db import Database
from nge_trader.services.metrics import set_metric, inc_metric_labeled
import pandas as pd


def main() -> None:
    svc = AppService()
    db = Database()
    n = Notifier()

    # Reconciliar órdenes
    diffs_orders = svc.reconcile_state(resolve=True)
    missing_db = len(diffs_orders.get("missing_in_db", []))
    missing_broker = len(diffs_orders.get("missing_in_broker", []))
    status_mis = len(diffs_orders.get("status_mismatches", []))
    try:
        set_metric("reconcile_missing_in_db", float(missing_db))
        set_metric("reconcile_missing_in_broker", float(missing_broker))
        set_metric("reconcile_status_mismatch", float(status_mis))
        if missing_db:
            inc_metric_labeled("reconciliation_runs_total", 1.0, {"result": "missing_in_db"})
        if missing_broker:
            inc_metric_labeled("reconciliation_runs_total", 1.0, {"result": "missing_in_broker"})
    except Exception:
        pass

    # Reconciliar posiciones/saldos
    diffs_pb = svc.reconcile_positions_balances(resolve=True)
    pos_count = len(diffs_pb.get("positions", []))
    bal_count = len(diffs_pb.get("balances", []))

    msg = (
        f"Reconciliación completada. Ordenes: +DB={missing_db}, -BRK={missing_broker}, ST_MIS={status_mis}. "
        f"Posiciones={pos_count}, BalSnaps={bal_count}."
    )
    try:
        n.send(msg)
        # Alerta por email si hay diferencias
        if missing_db or missing_broker:
            n.send_email_with_attachment(
                "Alerta de Reconciliación",
                msg,
                file_path=str(db.DB_PATH) if hasattr(db, "DB_PATH") else "data/app.db",
            )
        # Registrar en logs estructurados
        db.append_log_json("INFO", {"metric": "reconcile_result", "missing_in_db": missing_db, "missing_in_broker": missing_broker, "status_mismatch": status_mis, "positions": pos_count, "balances": bal_count}, pd.Timestamp.utcnow().isoformat())
    except Exception:
        pass
    print(msg)
    sys.exit(0)


if __name__ == "__main__":
    main()


