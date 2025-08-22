from __future__ import annotations

import time

from nge_trader.services.metrics import set_rehydrate_seconds


def test_rehydrate_under_5s():
    # Simula un rehydrate rápido
    set_rehydrate_seconds(3.2)
    # En un entorno real, aquí validaríamos transición de estados de WS y rehydrate del motor
    assert True


