from __future__ import annotations

import os
import pytest

from nge_trader.services.oms import place_market_order
from nge_trader.services.app_service import AppService
from nge_trader.repository.db import Database


class _OkBroker:
    def place_order(self, *args, **kwargs):  # noqa: ANN002, ANN003
        return {"id": "IDOK"}


def test_strategy_budget_blocks_when_limit_reached(monkeypatch):
    app = AppService()
    app.broker = _OkBroker()
    db = Database()
    # Simular que ya enviamos max_trades_per_day_per_strategy
    from nge_trader.config.settings import Settings
    s = Settings()
    # Forzar contador
    try:
        # Incrementar por encima del límite
        for _ in range(int(s.max_trades_per_day_per_strategy or 50)):
            db.inc_strategy_orders_sent_today(s.strategy_id, 1)
    except Exception:
        pass
    with pytest.raises(RuntimeError):
        place_market_order(app.broker, "TEST", "buy", 1.0)


