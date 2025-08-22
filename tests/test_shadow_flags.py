from __future__ import annotations

from nge_trader.services.strategy_flags import set_shadow, is_shadow


def test_shadow_gate():
    set_shadow("mean_rev", True)
    assert is_shadow("mean_rev") is True


