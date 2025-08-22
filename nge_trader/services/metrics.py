from __future__ import annotations

import math
from typing import Iterable, Dict, Deque
from threading import Lock
from collections import deque
try:
    from prometheus_client import Gauge  # type: ignore
except Exception:  # pragma: no cover
    Gauge = None  # type: ignore

import numpy as np
import pandas as pd


def compute_sharpe(returns: Iterable[float], risk_free_rate: float = 0.0) -> float:
    series = pd.Series(list(returns))
    if series.empty or series.std(ddof=0) == 0:
        return float("nan")
    excess = series - risk_free_rate / max(len(series), 252)
    return float(np.sqrt(252) * excess.mean() / series.std(ddof=0))


def compute_sortino(returns: Iterable[float], risk_free_rate: float = 0.0) -> float:
    series = pd.Series(list(returns))
    if series.empty:
        return float("nan")
    downside = series[series < 0]
    denom = downside.std(ddof=0)
    if denom == 0 or math.isnan(denom):
        return float("nan")
    excess = series - risk_free_rate / max(len(series), 252)
    return float(np.sqrt(252) * excess.mean() / denom)


def compute_max_drawdown(equity_curve: Iterable[float]) -> float:
    arr = np.array(list(equity_curve), dtype=float)
    if arr.size == 0:
        return float("nan")
    peaks = np.maximum.accumulate(arr)
    drawdowns = (arr - peaks) / peaks
    return float(drawdowns.min())


def compute_win_rate(returns: Iterable[float]) -> float:
    series = pd.Series(list(returns))
    if series.empty:
        return float("nan")
    wins = (series > 0).sum()
    return float(wins / len(series))


def compute_skewness(returns: Iterable[float]) -> float:
    series = pd.Series(list(returns))
    if series.empty:
        return float("nan")
    return float(series.skew())


def compute_kurtosis(returns: Iterable[float]) -> float:
    series = pd.Series(list(returns))
    if series.empty:
        return float("nan")
    # pandas returns Fisher definition (excess kurtosis)
    return float(series.kurt())

def compute_profit_factor(returns: Iterable[float]) -> float:
    series = pd.Series(list(returns))
    if series.empty:
        return float("nan")
    gains = series[series > 0].sum()
    losses = -series[series < 0].sum()
    denom = float(losses)
    if denom <= 0:
        return float("nan")
    return float(gains / denom)


def compute_drawdown_series(equity_curve: Iterable[float]) -> pd.Series:
    arr = np.array(list(equity_curve), dtype=float)
    if arr.size == 0:
        return pd.Series(dtype=float)
    peaks = np.maximum.accumulate(arr)
    dd = (arr - peaks) / peaks
    return pd.Series(dd)


def compute_calmar(equity_curve: Iterable[float]) -> float:
    arr = np.array(list(equity_curve), dtype=float)
    if arr.size < 2:
        return float("nan")
    total_ret = arr[-1] / arr[0] - 1.0
    ann_ret = (1.0 + total_ret) ** (252.0 / max(len(arr), 1)) - 1.0
    max_dd = abs(compute_max_drawdown(arr))
    if max_dd <= 0:
        return float("nan")
    return float(ann_ret / max_dd)



# ====== Export simple de métricas (stub Prometheus text format) ======
_METRICS_STORE: dict[str, float] = {}
_SAMPLES: list[tuple[str, dict[str, str], float]] = []


def set_metric(key: str, value: float) -> None:
    _METRICS_STORE[key] = float(value)


def inc_metric(key: str, value: float = 1.0) -> None:
    _METRICS_STORE[key] = float(_METRICS_STORE.get(key, 0.0) + value)


def export_metrics_text() -> str:
    lines: list[str] = []
    # Gauges/counters simples
    for k, v in sorted(_METRICS_STORE.items()):
        safe = k.replace(" ", "_")
        lines.append(f"{safe} {v}")
    # Muestras con etiquetas
    for name, labels, val in list(_SAMPLES):
        safe = name.replace(" ", "_")
        if labels:
            lbl = ",".join([f"{k}={repr(str(v)).strip('\'')}" for k, v in labels.items()])
            lines.append(f"{safe}{{{lbl}}} {val}")
        else:
            lines.append(f"{safe} {val}")
    # Exportar métricas de rate limit conocidas si están en store
    # Ejemplo: rate_limit_broker_remaining/capacity, rate_limit_strategy_remaining/capacity
    for name in [
        "rate_limit_broker_remaining",
        "rate_limit_broker_capacity",
        "rate_limit_strategy_remaining",
        "rate_limit_strategy_capacity",
    ]:
        if name in _METRICS_STORE:
            lines.append(f"{name} {_METRICS_STORE[name]}")
    # Histograms
    with _REG_LOCK:
        for name, hist in _HISTOGRAMS.items():
            for le, count in hist["buckets"].items():
                le_str = "+Inf" if le == float("inf") else (str(int(le)) if float(le).is_integer() else str(le))
                lines.append(f"{name}_bucket{{le=\"{le_str}\"}} {int(count)}")
            lines.append(f"{name}_count {int(hist['count'])}")
            lines.append(f"{name}_sum {float(hist['sum'])}")
        # WS uptime/reconnects
        for ws_name, st in _WS_STATES.items():
            now = pd.Timestamp.now().timestamp()
            uptime = float(st["uptime"] + ((now - st["last_start"]) if st["connected"] and st["last_start"] else 0.0))
            lines.append(f"ws_uptime_seconds{{ws=\"{ws_name}\"}} {uptime}")
            lines.append(f"ws_reconnects_total{{ws=\"{ws_name}\"}} {int(st['reconnects'])}")
        # Percentiles (export como gauges)
        for name, series in _SERIES.items():
            vals = list(series)
            if vals:
                p50 = float(np.percentile(vals, 50))
                p95 = float(np.percentile(vals, 95))
                lines.append(f"{name}_p50 {p50}")
                lines.append(f"{name}_p95 {p95}")
    return "\n".join(lines) + "\n"


# ====== Registro avanzado (histogramas, series, ws states) ======
_REG_LOCK = Lock()
_HISTOGRAMS: Dict[str, Dict] = {}
_SERIES: Dict[str, Deque[float]] = {}
_WS_STATES: Dict[str, Dict[str, float | bool]] = {}


def observe_latency(metric_name: str, value_ms: float, buckets: Iterable[float] | None = None, series_window: int = 500) -> None:
    bks = list(buckets or [50, 100, 200, 300, 500, 1000, float("inf")])
    with _REG_LOCK:
        if metric_name not in _HISTOGRAMS:
            _HISTOGRAMS[metric_name] = {"buckets": {bk: 0 for bk in bks}, "sum": 0.0, "count": 0}
        hist = _HISTOGRAMS[metric_name]
        # incrementar bucket correspondiente
        for bk in bks:
            if value_ms <= bk:
                hist["buckets"][bk] = int(hist["buckets"].get(bk, 0)) + 1
                break
        hist["sum"] = float(hist["sum"]) + float(value_ms)
        hist["count"] = int(hist["count"]) + 1
        # series para percentiles
        if metric_name not in _SERIES:
            _SERIES[metric_name] = deque(maxlen=int(series_window))
        _SERIES[metric_name].append(float(value_ms))


def observe_series(metric_name: str, value: float, window: int = 500) -> None:
    """Registra un valor escalar en una serie para exportar p50/p95 (como latency)."""
    with _REG_LOCK:
        if metric_name not in _SERIES:
            _SERIES[metric_name] = deque(maxlen=int(window))
        _SERIES[metric_name].append(float(value))


def update_ws_state(name: str, connected: bool) -> None:
    with _REG_LOCK:
        st = _WS_STATES.get(name)
        now = pd.Timestamp.now().timestamp()
        if st is None:
            st = {"connected": False, "last_start": None, "uptime": 0.0, "reconnects": 0}
            _WS_STATES[name] = st
        if connected:
            if not st["connected"]:
                st["connected"] = True
                st["last_start"] = now
                st["reconnects"] = int(st["reconnects"]) + 1
        else:
            if st["connected"]:
                # acumular uptime
                if st["last_start"]:
                    st["uptime"] = float(st["uptime"]) + (now - float(st["last_start"]))
                st["connected"] = False
                st["last_start"] = None


def get_ws_states_snapshot() -> Dict[str, Dict[str, float | bool]]:
    """Devuelve copia de estados WS (connected, uptime acumulado, reconnects, last_start)."""
    with _REG_LOCK:
        out: Dict[str, Dict[str, float | bool]] = {}
        for k, v in _WS_STATES.items():
            out[k] = dict(v)
        return out


def inc_metric_labeled(name: str, value: float, labels: dict[str, str] | None = None) -> None:
    _SAMPLES.append((str(name), dict(labels or {}), float(value)))


def set_metric_labeled(name: str, value: float, labels: dict[str, str] | None = None) -> None:
    # sobrescribir añadiendo última muestra (Prometheus recogerá última)
    _SAMPLES.append((str(name), dict(labels or {}), float(value)))


def get_series_percentile(metric_name: str, percentile: float) -> float:
    """Devuelve el percentil solicitado de una serie registrada (si existe)."""
    try:
        vals = list(_SERIES.get(metric_name, []))
        if not vals:
            return float("nan")
        import numpy as _np
        return float(_np.percentile(vals, float(percentile)))
    except Exception:
        return float("nan")


# ===== Métrica rehydrate_seconds (si prometheus_client disponible) =====
_rehydrate_gauge = Gauge("rehydrate_seconds", "Tiempo de rehydrate (s)") if Gauge else None  # type: ignore


def set_rehydrate_seconds(value_seconds: float) -> None:
    if _rehydrate_gauge is not None:
        try:
            _rehydrate_gauge.set(float(value_seconds))  # type: ignore[attr-defined]
        except Exception:
            pass
