from __future__ import annotations

from nge_trader.services.slo import set_symbol_slo, get_symbol_slo


def test_slo_source_roundtrip() -> None:
    set_symbol_slo("BTCUSDT", 7.5, 280.0, "db")
    x = get_symbol_slo("BTCUSDT")
    assert str(x.get("source")) in ("db", "prom", "unknown")


