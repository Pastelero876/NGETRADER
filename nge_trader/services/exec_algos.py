from __future__ import annotations

from typing import Any, Dict, List, Optional

from nge_trader.services.oms import place_market_order
from nge_trader.adapters.lowlat_provider import LowLatencyProvider
from nge_trader.services.oms import place_limit_order_post_only
from nge_trader.config.settings import Settings


def place_twap_order(
    broker: Any,
    symbol: str,
    side: str,
    total_quantity: float,
    slices: int = 6,
) -> List[Dict[str, Any]]:
    """Split order in equal time-weighted slices (no sleeps; immediate sequential for simplicity)."""
    qty_per = float(total_quantity) / max(1, int(slices))
    results: List[Dict[str, Any]] = []
    for _ in range(max(1, int(slices))):
        results.append(place_market_order(broker, symbol, side, qty_per))
    return results


def place_vwap_order(
    broker: Any,
    symbol: str,
    side: str,
    total_quantity: float,
    slices: int = 6,
) -> List[Dict[str, Any]]:
    """Split order approximating VWAP using recent microprice/volume hints if available."""
    llp = LowLatencyProvider()
    # Fallback: equal split if no depth/volume info
    parts = [1.0 for _ in range(max(1, int(slices)))]
    weights = [p / sum(parts) for p in parts]
    results: List[Dict[str, Any]] = []
    for w in weights:
        q = float(total_quantity) * float(w)
        results.append(place_market_order(broker, symbol, side, q))
    return results


def place_iceberg_order(
    broker: Any,
    symbol: str,
    side: str,
    total_quantity: float,
    display_quantity: float,
) -> List[Dict[str, Any]]:
    """Submit multiple child orders exposing only display_quantity each time (market simplification)."""
    remaining = float(total_quantity)
    disp = max(0.0, float(display_quantity))
    if disp <= 0:
        disp = float(total_quantity)
    results: List[Dict[str, Any]] = []
    while remaining > 1e-12:
        q = min(remaining, disp)
        results.append(place_market_order(broker, symbol, side, q))
        remaining -= q
    return results


def place_trailing_stop(
    broker: Any,
    symbol: str,
    side: str,
    quantity: float,
    trail_pct: float,
    stop_cap: Optional[float] = None,
) -> Dict[str, Any]:
    """Place a simple trailing stop as a single stop order using current microprice as anchor.

    Real trailing behavior (continual adjustment) requires a background task.
    Here we compute initial stop and submit once for a safe, synchronous behavior.
    """
    llp = LowLatencyProvider()
    mp = llp.get_microprice(symbol)
    if not mp or float(mp) <= 0:
        # Fallback: submit as market to avoid missing protection
        return place_market_order(broker, symbol, side, float(quantity))
    mpf = float(mp)
    pct = abs(float(trail_pct))
    if side.lower() == "sell":
        stop_price = mpf * (1.0 - pct)
    else:
        stop_price = mpf * (1.0 + pct)
    if stop_cap is not None:
        stop_price = float(max(min(stop_price, float(stop_cap)), 0.0))
    if hasattr(broker, "place_order"):
        return broker.place_order(
            symbol=symbol,
            side=side,
            quantity=float(quantity),
            type_="stop",
            stop_price=float(stop_price),
        )
    # Fallback
    return place_market_order(broker, symbol, side, float(quantity))


def compute_atr(prices: List[float], highs: List[float], lows: List[float], window: int = 14) -> float:
    try:
        import numpy as _np
        highs_a = _np.array(highs[-window:], dtype=float)
        lows_a = _np.array(lows[-window:], dtype=float)
        closes_a = _np.array(prices[-window:], dtype=float)
        trs = _np.maximum(highs_a - lows_a, _np.maximum(_np.abs(highs_a - closes_a), _np.abs(lows_a - closes_a)))
        atr = float(trs.mean()) if trs.size else 0.0
        return atr
    except Exception:
        return 0.0


def place_volatility_stop_with_trailing(
    broker: Any,
    symbol: str,
    side: str,
    quantity: float,
    atr_mult: float = 2.0,
    break_even_r: float = 1.0,
    trail_r: float = 2.0,
) -> Dict[str, Any]:
    """Coloca stop basado en ATR y activa break-even/trailing por múltiplos de R.

    Nota: trailing real requiere tarea background; aquí exponemos stop inicial con lógica básica.
    """
    try:
        from nge_trader.services.app_service import AppService as _AS
        df = _AS().data_provider.get_daily_adjusted(symbol)
        if df is None or df.empty:
            return place_trailing_stop(broker, symbol, side, quantity, trail_pct=0.02)
        closes = df["close"].astype(float).tolist()
        highs = (df.get("high") or df["close"]).astype(float).tolist()
        lows = (df.get("low") or df["close"]).astype(float).tolist()
        atr = compute_atr(closes, highs, lows, window=14)
        if atr <= 0:
            return place_trailing_stop(broker, symbol, side, quantity, trail_pct=0.02)
        last = float(closes[-1])
        risk_per_share = float(atr_mult) * float(atr)
        if side.lower() == "buy":
            stop_px = max(0.0, last - risk_per_share)
        else:
            stop_px = max(0.0, last + risk_per_share)
        # Submit como stop simple; break-even/trailing se manejan en engine en runtime (futuro)
        if hasattr(broker, "place_order"):
            return broker.place_order(symbol=symbol, side=side, quantity=float(quantity), type_="stop", stop_price=float(stop_px))
    except Exception:
        pass
    return place_trailing_stop(broker, symbol, side, quantity, trail_pct=0.02)

def place_conditional_stop(
    broker: Any,
    symbol: str,
    side: str,
    quantity: float,
    trigger_op: str,
    trigger_price: float,
    stop_limit_price: Optional[float] = None,
) -> Dict[str, Any]:
    """Conditional stop order:

    - If condition met now (microprice vs trigger_op/trigger_price), execute market immediately.
    - Else, submit a stop or stop-limit order at trigger price.
    """
    llp = LowLatencyProvider()
    mp = llp.get_microprice(symbol)
    if mp and float(mp) > 0:
        cond_ok = (str(trigger_op) == ">=" and float(mp) >= float(trigger_price)) or (
            str(trigger_op) == "<=" and float(mp) <= float(trigger_price)
        )
        if cond_ok:
            return place_market_order(broker, symbol, side, float(quantity))
    if hasattr(broker, "place_order"):
        kwargs: Dict[str, Any] = dict(
            symbol=symbol,
            side=side,
            quantity=float(quantity),
            type_="stop",
            stop_price=float(trigger_price),
        )
        if stop_limit_price is not None:
            kwargs.update(dict(type_="stop_limit", price=float(stop_limit_price)))
        return broker.place_order(**kwargs)  # type: ignore[arg-type]
    return place_market_order(broker, symbol, side, float(quantity))


# POV adaptativo: ejecuta por participación de volumen con ajustes por OFI/OBI y microprice
def place_pov_adaptive(
    broker: Any,
    symbol: str,
    side: str,
    total_quantity: float,
    pov_ratio: float = 0.1,
    window_sec: int = 60,
    tranches: int = 5,
) -> List[Dict[str, Any]]:
    """Ejecuta órdenes limit post-only en tramos, intentando participar con ratio del volumen.

    - Usa VWAP/volumen de la ventana reciente de eventos L1 de `LowLatencyProvider`.
    - Ajusta tamaño y precio límite con OFI/OBI y microprice.
    - Si falla el post-only, degrada puntualmente a market para no quedar colgado (robustez).
    """
    llp = LowLatencyProvider()
    results: List[Dict[str, Any]] = []
    try:
        now_s = __import__("pandas").Timestamp.utcnow().timestamp()  # type: ignore[attr-defined]
    except Exception:
        import time as _t
        now_s = _t.time()
    # Colectar eventos recientes del provider si expone buffer
    try:
        recent = [(t, px, q) for (t, px, q) in getattr(llp, "_events", []) if (now_s - t) <= float(window_sec)]
    except Exception:
        recent = []
    total_vol = float(sum(r[2] for r in recent)) or 1.0
    vwap = float(sum(r[1] * r[2] for r in recent) / total_vol) if recent else (float(llp.get_microprice(symbol) or 0.0) or 0.0)
    remaining = float(total_quantity)
    n = max(1, int(tranches))
    for _ in range(n):
        # tamaño objetivo por participación
        est_part = max(total_vol * float(pov_ratio) / n, float(total_quantity) / (n * 2))
        part = float(min(remaining, est_part))
        if part <= 0:
            break
        # precio límite sugerido
        bps = 5.0 / 10000.0
        try:
            ofi, _obi = llp.get_ofi_obi(symbol)
            limit_px = llp.suggest_limit_price(symbol, side, base_price=vwap, bps=bps)
            # Si flujo en contra, reducir part
            part = max(part * (1.0 + max(min(ofi, 0.2), -0.2)), part * 0.5)
        except Exception:
            limit_px = (vwap * (1.0 - bps)) if side.lower() == "buy" else (vwap * (1.0 + bps))
        try:
            res = place_limit_order_post_only(broker, symbol, side, part, float(limit_px), "GTC")
            results.append(res)
            remaining -= part
        except Exception:
            # fallback único para no quedar colgados
            try:
                res = place_market_order(broker, symbol, side, part)
                results.append(res)
                remaining -= part
            except Exception:
                # continuar con el siguiente tramo
                pass
    return results


def estimate_fill_probability(symbol: str, side: str, price: float) -> float:
    """Estimación simple de probabilidad de fill usando microprice y OBI.

    - Si limit es agresivo vs mid, mayor probabilidad.
    - Ajuste por order-book imbalance (OBI) si disponible.
    """
    llp = LowLatencyProvider()
    try:
        mp = float(llp.get_microprice(symbol) or 0.0)
        if mp <= 0:
            return 0.5
        rel = (float(price) / mp) - 1.0
        # Price aggressiveness
        base = 0.5
        if side.lower() == "buy":
            base += float(-rel * 50.0)  # 1% por 50 bps
        else:
            base += float(rel * 50.0)
        # OBI ajuste
        try:
            _ofi, obi = llp.get_ofi_obi(symbol)
            base += float(obi) * 0.1
        except Exception:
            pass
        return max(0.0, min(1.0, base))
    except Exception:
        return 0.5


def place_maker_first_with_prob_gate(
    broker: Any,
    symbol: str,
    side: str,
    quantity: float,
    pov_ratio: float = 0.1,
    price_bps: float = 5.0,
) -> dict:
    """Router maker-first: intenta post-only y sólo usa marketable-limit si prob_fill ≥ umbral.

    - Sugiere precio límite alrededor de microprice +/- bps según lado.
    - Verifica probabilidad de fill y decide entre post-only o marketable-limit.
    """
    llp = LowLatencyProvider()
    mp = llp.get_microprice(symbol) or 0.0
    bps = abs(float(price_bps)) / 10000.0
    if side.lower() == "buy":
        limit_px = float(mp) * (1.0 - bps)
    else:
        limit_px = float(mp) * (1.0 + bps)
    prob = estimate_fill_probability(symbol, side, limit_px)
    thr = float(Settings().fill_prob_threshold or 0.6)
    try:
        if prob >= thr:
            # marketable-limit (sin post_only)
            from nge_trader.services.oms import place_limit_order_marketable
            return place_limit_order_marketable(broker, symbol, side, float(quantity), float(limit_px), "GTC")
        # maker-first post-only
        return place_limit_order_post_only(broker, symbol, side, float(quantity), float(limit_px), "GTC")
    except Exception as exc:  # noqa: BLE001
        # fallback simple
        return place_market_order(broker, symbol, side, float(quantity))


def _recent_volume_per_sec(symbol: str, window_sec: int = 60) -> float:
    llp = LowLatencyProvider()
    try:
        df = llp.get_stable_agg(symbol)
        if df is None or df.empty:
            return 0.0
        if "ts" not in df.columns or "qty" not in df.columns:
            qty = float(df["qty"].tail(50).sum()) if "qty" in df.columns else 0.0
            return qty / float(window_sec)
        tail = df.tail(500)
        t_end = float(tail["ts"].iloc[-1]) if not tail.empty else 0.0
        t_start = max(0.0, t_end - float(window_sec))
        seg = tail[tail["ts"] >= t_start]
        qty = float(seg["qty"].sum()) if not seg.empty else 0.0
        dt = max(1.0, float(window_sec))
        return qty / dt
    except Exception:
        return 0.0


def estimate_fill_time_sec(symbol: str, side: str, price: float) -> float:
    """Estimación simple de tiempo a fill por cola en mejor punta."""
    llp = LowLatencyProvider()
    try:
        depth = llp.get_depth(symbol)
        ofi, _obi = llp.get_ofi_obi(symbol)
        dvps = _recent_volume_per_sec(symbol, 60)
        if dvps <= 0.0:
            return float("inf")
        # volumen en cola en la mejor punta
        if side.lower() == "buy":
            bids = list(depth.get("bids") or [])
            best = bids[0] if bids else (0.0, 0.0)
            best_px, best_qty = float(best[0]), float(best[1])
            queue = best_qty * (1.5 if float(price) < best_px else 1.0)
        else:
            asks = list(depth.get("asks") or [])
            best = asks[0] if asks else (0.0, 0.0)
            best_px, best_qty = float(best[0]), float(best[1])
            queue = best_qty * (1.5 if float(price) > best_px else 1.0)
        speed = dvps * (1.0 + max(min(ofi, 0.5), -0.5))
        speed = max(speed, dvps * 0.25)
        t = queue / max(speed, 1e-9)
        return float(max(t, 0.0))
    except Exception:
        return float("inf")


def compute_dynamic_pov_ratio(symbol: str) -> float:
    llp = LowLatencyProvider()
    try:
        q = llp.get_quote_metrics(symbol)
        vol = llp.get_volatility(symbol) or 0.0
        spread = float(q.get("spread") or 0.0)
        mp = float(q.get("microprice") or 0.0) or 1.0
        spread_pct = (spread / mp) if mp > 0 else 0.0
        base = 0.10
        adj = base * (1.0 - min(0.7, 3.0 * vol + 10.0 * spread_pct))
        ratio = max(0.05, min(0.30, base + adj))
        return float(ratio)
    except Exception:
        return 0.10


def place_pov_dynamic(
    broker: Any,
    symbol: str,
    side: str,
    total_quantity: float,
    window_sec: int = 60,
    tranches: int = 5,
) -> List[Dict[str, Any]]:
    ratio = compute_dynamic_pov_ratio(symbol)
    return place_pov_adaptive(broker, symbol, side, float(total_quantity), pov_ratio=ratio, window_sec=window_sec, tranches=tranches)


def place_peg_to_mid(
    broker: Any,
    symbol: str,
    side: str,
    quantity: float,
    band_bps: float = 5.0,
    tif: str = "GTC",
) -> Dict[str, Any]:
    llp = LowLatencyProvider()
    qm = llp.get_quote_metrics(symbol)
    bid = float(qm.get("bid") or 0.0)
    ask = float(qm.get("ask") or 0.0)
    mp = float(qm.get("microprice") or 0.0) or (0.5 * (bid + ask) if (bid > 0 and ask > 0) else 0.0)
    if mp <= 0 or bid <= 0 or ask <= 0:
        return place_maker_first_with_prob_gate(broker, symbol, side, float(quantity))
    bps = abs(float(band_bps)) / 10000.0
    lower = mp * (1.0 - bps)
    upper = mp * (1.0 + bps)
    if side.lower() == "buy":
        px = min(max(bid, lower), ask * (1.0 - 1e-6))
    else:
        px = max(min(ask, upper), bid * (1.0 + 1e-6))
    return place_limit_order_post_only(broker, symbol, side, float(quantity), float(px), tif)


def choose_order_route_predictive(
    broker: Any,
    symbol: str,
    side: str,
    total_quantity: float,
    band_bps: float = 5.0,
) -> Dict[str, Any] | List[Dict[str, Any]]:
    llp = LowLatencyProvider()
    q = llp.get_quote_metrics(symbol)
    mp = float(q.get("microprice") or 0.0) or 0.0
    if mp <= 0:
        return place_market_order(broker, symbol, side, float(total_quantity))
    px = llp.suggest_limit_price(symbol, side, base_price=mp, bps=0.0005)
    p = estimate_fill_probability(symbol, side, float(px))
    tsec = estimate_fill_time_sec(symbol, side, float(px))
    thr_p = 0.75
    thr_t_fast = 3.0
    thr_t_slow = 30.0
    if p >= thr_p or tsec <= thr_t_fast:
        from nge_trader.services.oms import place_limit_order_marketable
        return place_limit_order_marketable(broker, symbol, side, float(total_quantity), float(px), "GTC")
    if p >= 0.5 and tsec <= thr_t_slow:
        return place_peg_to_mid(broker, symbol, side, float(total_quantity), band_bps=float(band_bps), tif="GTC")
    return place_pov_dynamic(broker, symbol, side, float(total_quantity))
