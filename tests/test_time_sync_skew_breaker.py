from __future__ import annotations

import time

from nge_trader.services.time_sync import estimate_skew_ms
from nge_trader.services import metrics as metrics_mod


def test_time_skew_metric_set(monkeypatch):
    def fake_ntp():
        return time.time() - 0.2  # 200ms atraso

    monkeypatch.setattr("nge_trader.services.time_sync.get_ntp_time", lambda **_: fake_ntp())
    skew = estimate_skew_ms(server="dummy")
    assert skew >= 199.0
    assert metrics_mod._METRICS_STORE.get("time_skew_ms", 0.0) >= 199.0


def test_ntp_poll_failure_increments(monkeypatch):
    before = float(metrics_mod._METRICS_STORE.get("ntp_poll_failures_total", 0.0))
    monkeypatch.setattr("nge_trader.services.time_sync.get_ntp_time", lambda **_: (_ for _ in ()).throw(RuntimeError("ntp error")))
    val = estimate_skew_ms(server="dummy")
    after = float(metrics_mod._METRICS_STORE.get("ntp_poll_failures_total", 0.0))
    assert val == 0.0
    assert after >= before + 1.0

import pytest


class _NoopBroker:
    def place_order(self, *args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("should not be called when skew breaker triggers")


def test_skew_breaker_blocks_orders(monkeypatch):
    # Forzar skew alto y política de reintentos activa
    monkeypatch.setenv("ENABLE_RETRY_POLICY", "1")
    monkeypatch.setenv("MAX_TIME_SKEW_MS", "50")
    from nge_trader.services import oms

    monkeypatch.setattr(oms, "estimate_skew_ms", lambda server=None: 1000.0)

    with pytest.raises(RuntimeError):
        oms.place_market_order(_NoopBroker(), symbol="TEST", side="buy", quantity=1.0)


