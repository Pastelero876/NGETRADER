from __future__ import annotations

from nge_trader.services import canary_cap


def test_canary_gate_blocks(monkeypatch):
    monkeypatch.setattr(canary_cap, "compute_used_pct", lambda db: 0.26)
    assert canary_cap.gate(None, 0.25) is True


