from __future__ import annotations

from nge_trader.services.oms import _quantize_step, quantize_quantity_price
from nge_trader.config.settings import Settings


def test_quantize_step_basic():
    assert _quantize_step(1.2345, 0.01) == 1.23
    assert _quantize_step(1.2, 0.05) == 1.2
    assert _quantize_step(1.24, 0.05) == 1.2


def test_quantize_quantity_price_per_symbol(monkeypatch):
    class S:
        qty_step = 0.0
        price_tick = 0.01
        lot_size_per_symbol = '{"BTCUSDT": 0.001}'

    monkeypatch.setattr("nge_trader.services.oms.Settings", lambda: S())
    q, p = quantize_quantity_price("BTCUSDT", 0.001234, 123.4567)
    assert abs(q - 0.001) < 1e-12
    assert abs(p - 123.45) < 1e-12


