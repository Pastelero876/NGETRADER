from __future__ import annotations

import pytest


class _FlakyBroker:
    def __init__(self) -> None:
        self.calls = 0

    def place_order(self, *args, **kwargs):  # noqa: ANN002, ANN003
        self.calls += 1
        raise RuntimeError("fail")


def test_slo_gate_blocks_after_errors(monkeypatch):
    # Forzar umbral bajo de error-rate
    monkeypatch.setenv("MAX_ERROR_RATE", "0.0")
    from nge_trader.services.oms import place_market_order

    broker = _FlakyBroker()
    # Primera llamada debe intentar y fallar
    with pytest.raises(RuntimeError):
        place_market_order(broker, symbol="TST", side="buy", quantity=1.0)
    # Segunda llamada debe ser bloqueada por SLO gate
    with pytest.raises(RuntimeError):
        place_market_order(broker, symbol="TST", side="buy", quantity=1.0)


