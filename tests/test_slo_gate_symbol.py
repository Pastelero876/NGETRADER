from __future__ import annotations

from nge_trader.services.slo import set_symbol_slo, get_symbol_slo


def test_slo_set_get() -> None:
    set_symbol_slo("BTCUSDT", 9.0, 250.0)
    slo = get_symbol_slo("BTCUSDT")
    assert slo["slip"] == 9.0 and slo["p95"] == 250.0



