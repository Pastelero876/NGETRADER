from __future__ import annotations

from math import isclose


def test_slip_formula_buy_sell() -> None:
    def slip(side: str, fill: float, ref: float) -> float:
        if side == "buy":
            return ((fill - ref) / ref) * 10000.0
        return ((ref - fill) / ref) * 10000.0

    assert isclose(slip("buy", 101.0, 100.0), 100.0, rel_tol=1e-6)
    assert isclose(slip("sell", 99.0, 100.0), 100.0, rel_tol=1e-6)
