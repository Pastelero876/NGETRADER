from __future__ import annotations

from types import SimpleNamespace
from nge_trader.services.shadow_recorder import ShadowEvent, write


def test_shadow_recorder_builds_insert(monkeypatch):
    executed = {}

    class DummyDB:
        def execute(self, q, params):  # type: ignore
            executed["params"] = params
            return None

    ev = ShadowEvent(
        symbol="BTCUSDT", ts=1700000000.0, side="buy", est_price=100.0, ref_px=99.5,
        est_slippage_bps=50.0, route_algo="TWAP", strategy="mean_rev", qty=0.01, meta={"shadow": True}
    )
    write(DummyDB(), ev)
    assert executed.get("params", {}).get("strategy") == "mean_rev"
    assert executed.get("params", {}).get("meta", {}).get("shadow") is True


