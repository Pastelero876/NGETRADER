from __future__ import annotations

from nge_trader.services.gates import is_trade_viable
from nge_trader.services import health as _H


def test_viability_ok() -> None:
    assert is_trade_viable(9.0, 4.0, 2.0)  # 9 >= 4+2+2


def test_viability_block() -> None:
    assert not is_trade_viable(6.0, 4.0, 2.0)


def test_preflight_blocks() -> None:
    # Forzar bloqueos simulando skew alto
    sk = _H.time_skew_ms()
    assert isinstance(sk, int)
    # Aceptación: skew>100 debería bloquear en endpoint, aquí verificamos helper
    from nge_trader.services.gates import time_skew_ok
    assert time_skew_ok(200, 100) is False


def test_costs_gate() -> None:
    # edge < fees+hs+buffer bloquea
    assert is_trade_viable(7.9, 4.0, 2.0, buffer_bps=2.0) is False


