from __future__ import annotations

import time

from nge_trader.adapters.resilient import ResilientBroker


class _Flaky:
    def __init__(self, fail_times: int = 2) -> None:
        self._fails_left = int(fail_times)

    def op(self, x: int) -> int:
        if self._fails_left > 0:
            self._fails_left -= 1
            raise RuntimeError("transient")
        return x * 2


class _Fallback:
    def op(self, x: int) -> int:  # noqa: D401
        # fallback deterministic
        return x + 1


def test_retry_policy_success_after_retries():
    primary = _Flaky(fail_times=2)
    fallback = _Fallback()
    rb = ResilientBroker(primary, fallback, max_attempts=3)
    rb._base_sleep = 0.0  # speed up test
    out = rb.op(5)
    assert out == 10


def test_retry_policy_fallback_after_exhausted():
    primary = _Flaky(fail_times=10)
    fallback = _Fallback()
    rb = ResilientBroker(primary, fallback, max_attempts=2)
    rb._base_sleep = 0.0
    out = rb.op(7)
    # fallback result
    assert out == 8

class _FailingPrimary:
    def __init__(self) -> None:
        self.calls = 0

    def place_order(self, *args, **kwargs):  # noqa: ANN002, ANN003
        self.calls += 1
        raise RuntimeError("primary failed")


class _FallbackOK:
    def __init__(self) -> None:
        self.calls = 0

    def place_order(self, *args, **kwargs):  # noqa: ANN002, ANN003
        self.calls += 1
        return {"status": "ok", "id": "fallback-1"}


def test_resilient_broker_fallback():
    from nge_trader.adapters.resilient import ResilientBroker

    rb = ResilientBroker(primary_broker=_FailingPrimary(), fallback_broker=_FallbackOK(), max_attempts=2)
    res = rb.place_order(symbol="TEST", side="buy", quantity=1.0, type_="market")
    assert isinstance(res, dict) and res.get("status") == "ok"


