from __future__ import annotations


def test_audit_extra_keys_exact() -> None:
    # Minimal construction of audit dict
    audit_extra = {
        "correlation_id": "abc",
        "model_used": "pinned",
        "model_id": "m1",
        "canary": False,
        "edge_bps": 1.0,
        "costs_bps": 3.0,
        "route_algo": "TWAP",
    }
    assert set(audit_extra.keys()) == {"correlation_id", "model_used", "model_id", "canary", "edge_bps", "costs_bps", "route_algo"}


