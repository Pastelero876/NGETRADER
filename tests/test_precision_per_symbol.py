from __future__ import annotations

from nge_trader.services.precision import round_price_qty


def test_round_price_qty():
    px, q = round_price_qty("BTCUSDT", 123.4567, 0.001234)
    assert round(px or 0, 2) == 123.45
    assert abs(q - 0.001234) <= 0.001234  # sin lot map definido, conserva qty


