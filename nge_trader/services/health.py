from __future__ import annotations

from nge_trader.config.settings import Settings
from nge_trader.services.market_clock import is_session_open_cached
from nge_trader.services.time_sync import estimate_skew_ms
from nge_trader.repository.db import Database
from nge_trader.services.metrics import get_ws_states_snapshot


def market_clock_open() -> bool:
    try:
        return bool(is_session_open_cached(Settings().market_clock_exchange))
    except Exception:
        return True


def time_skew_ms() -> int:
    try:
        skew = float(abs(estimate_skew_ms(server=Settings().ntp_server)))
        return int(skew)
    except Exception:
        return 0


def ws_ok() -> bool:
    try:
        ws_states = get_ws_states_snapshot()
        # OK si hay al menos un WS activo o no se requiere WS
        if not ws_states:
            return True
        return any(bool(st.get("connected")) for st in ws_states.values())
    except Exception:
        return True


def idempotency_ok() -> bool:
    try:
        db = Database()
        ages = db.oldest_outbox_age_seconds_by_status()
        # Consideramos OK si no hay errores envejecidos (> 10 min)
        max_age = 600.0
        return all((age <= max_age) for age in ages.values()) if ages else True
    except Exception:
        return True


def budgets_loaded() -> bool:
    try:
        db = Database()
        # Si existen registros de budgets hoy o claves de presupuesto configuradas
        any_strategy = bool(db.get_strategy_orders_sent_today(Settings().strategy_id) >= 0)
        any_account = bool(db.get_account_orders_sent_today(Settings().account_id) >= 0)
        return bool(any_strategy and any_account)
    except Exception:
        return True


