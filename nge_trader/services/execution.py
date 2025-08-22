from __future__ import annotations

import pandas as pd
from typing import Callable
import time
from collections import deque
from nge_trader.config.settings import Settings
from nge_trader.services.metrics import set_metric
from nge_trader.services.metrics import inc_metric


class ExecutionAlgos:
    """Modelos simples de ejecución para backtesting.

    Ajustan el precio teórico de ejecución usando columnas disponibles (open/high/low/close/volume).
    """

    @staticmethod
    def apply(price: float, side: float, row: pd.Series, algo: str | None, fallback_slippage: float) -> float:
        if not algo:
            return price * (1 + fallback_slippage if side > 0 else 1 - fallback_slippage)
        algo = algo.lower()
        try:
            if algo == "twap":
                o = float(row.get("open", price))
                c = float(row.get("close", price))
                twap = (o + c) / 2.0
                return twap
            if algo == "vwap":
                c = float(row.get("close", price))
                v = float(row.get("volume", 0.0))
                # sin intradía: aproximar vwap por close y reducir slippage a la mitad
                return c * (1 + (fallback_slippage * 0.5 if side > 0 else -fallback_slippage * 0.5))
            if algo == "pov":
                # participation of volume: usar precio y slippage reducido, asumiendo ejecución repartida
                return price * (1 + (fallback_slippage * 0.7 if side > 0 else -fallback_slippage * 0.7))
            if algo == "sniper":
                # intenta mejorar sobre close usando min(open,close) para buy y max para sell
                o = float(row.get("open", price))
                c = float(row.get("close", price))
                px = min(o, c) if side > 0 else max(o, c)
                return px * (1 + (fallback_slippage * 0.3 if side > 0 else -fallback_slippage * 0.3))
            if algo == "adaptive":
                # escoge entre twap/vwap según disponibilidad de volumen
                v = float(row.get("volume", 0.0))
                if v > 0:
                    c = float(row.get("close", price))
                    return c * (1 + (fallback_slippage * 0.5 if side > 0 else -fallback_slippage * 0.5))
                o = float(row.get("open", price))
                c = float(row.get("close", price))
                return (o + c) / 2.0
        except Exception:
            pass
        return price * (1 + fallback_slippage if side > 0 else 1 - fallback_slippage)


# ========= Gating por SLOs (error-rate/slippage) =========

class SLOGate:
    """Controla envío de órdenes según SLOs recientes.

    - error-rate: errores / intentos en ventana
    - slippage: usa métrica externa (e.g., rolling media) si está disponible
    """

    def __init__(self, window_seconds: int = 600) -> None:
        self.window_seconds = int(window_seconds)
        self._attempts: deque[float] = deque(maxlen=2000)
        self._errors: deque[float] = deque(maxlen=2000)

    def register_attempt(self) -> None:
        self._attempts.append(time.time())

    def register_error(self) -> None:
        self._errors.append(time.time())

    def can_send(self) -> bool:
        s = Settings()
        now = time.time()
        # depurar ventana
        self._attempts = deque([t for t in self._attempts if (now - t) <= self.window_seconds], maxlen=2000)
        self._errors = deque([t for t in self._errors if (now - t) <= self.window_seconds], maxlen=2000)
        attempts = max(len(self._attempts), 1)
        err_rate = len(self._errors) / attempts
        set_metric("slo_error_rate_recent", float(err_rate))
        return err_rate <= float(s.max_error_rate or 0.2)


# ========= Coalescing/colas de envío =========

class OrderCoalescer:
    """Agrupa señales por símbolo/side dentro de una ventana y consolida envíos.

    Uso:
      coalescer = OrderCoalescer(window_ms=150)
      coalescer.add(symbol, side, qty)
      # periódicamente
      results = coalescer.flush(route_fn)

    Donde route_fn(symbol, side, total_qty) -> dict
    """

    def __init__(self, window_ms: int = 150) -> None:
        self.window_ms = int(window_ms)
        self._q: dict[tuple[str, str], dict] = {}

    def add(self, symbol: str, side: str, quantity: float) -> None:
        key = (symbol.upper(), side.lower())
        now = time.time() * 1000.0
        cur = self._q.get(key)
        if cur is None:
            self._q[key] = {"qty": float(quantity), "first_ms": now, "last_ms": now, "count": 1}
        else:
            cur["qty"] = float(cur.get("qty", 0.0)) + float(quantity)
            cur["last_ms"] = now
            cur["count"] = int(cur.get("count", 0)) + 1

    def flush(self, route_fn: Callable[[str, str, float], dict]) -> list[dict]:
        out: list[dict] = []
        now = time.time() * 1000.0
        keys_to_del: list[tuple[str, str]] = []
        for key, info in self._q.items():
            if (now - float(info.get("first_ms", now))) >= float(self.window_ms):
                symbol, side = key
                qty = float(info.get("qty", 0.0))
                if qty <= 0:
                    keys_to_del.append(key)
                    continue
                try:
                    res = route_fn(symbol, side, qty)
                    out.append(res)
                    # métricas: coalescing
                    try:
                        inc_metric("orders_coalesced_total", float(info.get("count", 1) - 1))
                        set_metric("coalescing_queue_exit_rate", 1.0)
                        set_metric("queue_exit_rate", 1.0)
                    except Exception:
                        pass
                except Exception:
                    try:
                        set_metric("coalescing_queue_exit_rate", 0.0)
                        set_metric("queue_exit_rate", 0.0)
                    except Exception:
                        pass
                keys_to_del.append(key)
        for k in keys_to_del:
            self._q.pop(k, None)
        return out

