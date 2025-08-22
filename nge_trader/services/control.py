from __future__ import annotations

import os
from typing import List

from nge_trader.services.app_service import AppService
from nge_trader.services.oms import cancel_all_orders_by_symbol
from nge_trader.services.metrics import inc_metric, set_metric_labeled


def list_open_symbols() -> List[str]:
    """Devuelve símbolos con órdenes abiertas o posiciones (stub)."""
    # TODO: sustituir por consulta real a DB/OMS
    return ["BTCUSDT", "ETHUSDT"]


def cancel_all_for_symbol(symbol: str) -> dict:
    """Cancela todas las órdenes del símbolo contra el broker/OMS real."""
    svc = AppService()
    symu = symbol.upper()
    try:
        res = cancel_all_orders_by_symbol(svc.broker, symu)
        inc_metric("orders_canceled_total", 1.0)
        set_metric_labeled("orders_canceled_total", 1.0, {"symbol": symu})
        return {"symbol": symu, "canceled": True, "result": res}
    except Exception as exc:  # noqa: BLE001
        return {"symbol": symu, "canceled": False, "error": str(exc)}


def disarm_and_cancel_if_config() -> dict:
    did_cancel = False
    details: list[dict] = []
    if str(os.getenv("KILL_CANCEL_PENDING", "false")).lower() == "true":
        for sym in list_open_symbols():
            r = cancel_all_for_symbol(sym)
            details.append(r)
            did_cancel = True
    return {"did_cancel": bool(did_cancel), "details": details}


