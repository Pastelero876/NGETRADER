from __future__ import annotations

from nge_trader.services.metrics import set_rehydrate_seconds


def test_rehydrate_metric_set() -> None:
    set_rehydrate_seconds(1.23)


