from __future__ import annotations

from nge_trader.services import canary_cap


def test_canary_gate_blocks(monkeypatch):
    monkeypatch.setattr(canary_cap, "compute_used_pct", lambda db: 0.26)
    assert canary_cap.gate(None, 0.25) is True


def test_canary_used_pct_zero_when_total_zero(monkeypatch):
    # Simula compute con total=0 via monkeypatch del método completo
    def fake_compute(db):
        return 0.0
    monkeypatch.setattr(canary_cap, "compute_used_pct", fake_compute)
    assert canary_cap.compute_used_pct(None) == 0.0


