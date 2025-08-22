from __future__ import annotations

import functools
import random
import time
from typing import Any, Callable
from nge_trader.services.metrics import inc_metric_labeled, set_metric_labeled


class ResilientBroker:
    """Envoltura que añade backoff/reintentos y failover a uno o varios brokers de respaldo.

    - Compatible con firma anterior (un único `fallback_broker`).
    - Si todos fallan, propaga la última excepción.
    """

    def __init__(self, primary_broker: Any, fallback_broker: Any | None = None, max_attempts: int = 3, fallback_brokers: list[Any] | None = None) -> None:
        self._primary = primary_broker
        # Normalizar lista de fallbacks
        fbs: list[Any] = []
        if fallback_brokers:
            fbs.extend([fb for fb in fallback_brokers if fb is not None])
        if fallback_broker is not None:
            fbs.append(fallback_broker)
        self._fallbacks = list(fbs)
        self._max_attempts = max_attempts
        self._base_sleep = 0.25

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._primary, name)
        if not callable(attr):
            return attr

        fallback_chain: list[tuple[str, Callable[..., Any]]] = []
        for fb in self._fallbacks:
            if hasattr(fb, name):
                fb_attr = getattr(fb, name)
                if callable(fb_attr):
                    fb_name = getattr(fb, "__class__", type("B", (), {})).__name__
                    fallback_chain.append((fb_name, fb_attr))

        @functools.wraps(attr)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: Exception | None = None
            for attempt in range(self._max_attempts):
                try:
                    t0 = time.perf_counter()
                    out = attr(*args, **kwargs)
                    dt = (time.perf_counter() - t0) * 1000.0
                    try:
                        set_metric_labeled("broker_call_latency_ms", float(dt), {"method": name})
                    except Exception:
                        pass
                    return out
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    # Backoff exponencial con jitter
                    sleep_base = min((2 ** attempt) * self._base_sleep, 5.0)
                    jitter = random.uniform(0.5, 1.5)
                    time.sleep(sleep_base * jitter)
            # Cadena de fallbacks
            for fb_name, fb_attr in fallback_chain:
                try:
                    out = fb_attr(*args, **kwargs)
                    try:
                        inc_metric_labeled("broker_failover_total", 1.0, {"method": name, "broker": fb_name})
                    except Exception:
                        pass
                    return out
                except Exception:
                    continue
            if last_exc:
                raise last_exc
            raise RuntimeError("Operación fallida sin excepción registrada")

        return wrapper


