from __future__ import annotations

from collections import deque
from typing import Dict, List, Tuple, Optional

import pandas as pd

from nge_trader.repository.db import Database


def recompute_lot_accounting(symbol: Optional[str] = None, method: str = "FIFO") -> Dict[str, float]:
    """Recalcula P&L realized por símbolo usando FIFO o WA sobre fills.

    Escribe resultados en la tabla `trades` como resumen simple (append).
    Devuelve métricas agregadas.
    """
    db = Database()
    # Obtener fills recientes (simple: usamos todos; en producción filtrar por rango)
    fills = db.recent_fills(10000)
    if symbol:
        fills = [f for f in fills if (f.get("symbol") or "").upper() == symbol.upper()]
    # Agrupar por símbolo y ordenar
    by_sym: Dict[str, List[dict]] = {}
    for f in fills:
        by_sym.setdefault((f.get("symbol") or "").upper(), []).append(f)
    total_realized = 0.0
    count = 0
    for sym, fs in by_sym.items():
        fs_sorted = sorted(fs, key=lambda x: x.get("ts") or "")
        # Inventario
        inv: deque[Tuple[float, float]] = deque()
        realized = 0.0
        wa_qty = 0.0
        wa_cost = 0.0
        for f in fs_sorted:
            side = (f.get("side") or "").lower()
            qty = float(f.get("qty") or 0.0)
            px = float(f.get("price") or 0.0)
            fee = float(f.get("fee") or f.get("fees") or 0.0)
            if qty <= 0 or px <= 0:
                continue
            if method.upper() == "WA":
                if side == "buy":
                    wa_cost = (wa_cost * wa_qty + px * qty) / (wa_qty + qty) if (wa_qty + qty) > 0 else px
                    wa_qty += qty
                elif side == "sell":
                    take = min(wa_qty, qty)
                    realized += (px - wa_cost) * take
                    wa_qty = max(0.0, wa_qty - take)
                    realized -= fee
            else:  # FIFO
                if side == "buy":
                    inv.append((qty, px))
                elif side == "sell":
                    remain = qty
                    while remain > 1e-12 and inv:
                        q0, p0 = inv[0]
                        take = min(q0, remain)
                        realized += (px - p0) * take
                        q0 -= take
                        remain -= take
                        if q0 <= 1e-12:
                            inv.popleft()
                        else:
                            inv[0] = (q0, p0)
                    realized -= fee
        # Guardar resumen en trades como registro consolidado de realized
        if realized != 0.0:
            db.record_trade({
                "symbol": sym,
                "side": "sell",
                "qty": 0.0,
                "price": 0.0,
                "in_time": None,
                "out_time": pd.Timestamp.now(tz=pd.Timestamp.utcnow().tz).isoformat(),
                "fees": 0.0,
                "realized": realized,
            })
            total_realized += realized
            count += 1
    return {"symbols": float(len(by_sym)), "entries": float(count), "total_realized": float(total_realized)}


def _get_last_price(symbol: str, db: Database) -> Optional[float]:
    # Preferir microprice de low-lat provider
    try:
        from nge_trader.adapters.lowlat_provider import LowLatencyProvider  # noqa: WPS433
        mp = LowLatencyProvider().get_microprice(symbol)
        if mp and float(mp) > 0:
            return float(mp)
    except Exception:
        pass
    # Fallback: último fill
    try:
        fills = [f for f in db.recent_fills(50) if (f.get("symbol") or "").upper() == symbol.upper()]
        if fills:
            return float(fills[0].get("price") or 0.0)
    except Exception:
        return None
    return None


def compute_pnl_summary(symbol: Optional[str] = None, method: str = "FIFO") -> List[Dict[str, float]]:
    """Calcula P&L realizado y no realizado por símbolo.

    Devuelve lista con: symbol, qty_open, avg_cost, last_price, unrealized, realized_total, fees_total.
    """
    db = Database()
    fills = db.recent_fills(10000)
    if symbol:
        fills = [f for f in fills if (f.get("symbol") or "").upper() == symbol.upper()]
    by_sym: Dict[str, List[dict]] = {}
    for f in fills:
        by_sym.setdefault((f.get("symbol") or "").upper(), []).append(f)
    out: List[Dict[str, float]] = []
    for sym, fs in by_sym.items():
        fs_sorted = sorted(fs, key=lambda x: x.get("ts") or "")
        inv: deque[Tuple[float, float]] = deque()  # (qty, price)
        wa_qty = 0.0
        wa_cost = 0.0
        realized = 0.0
        fees_total = 0.0
        for f in fs_sorted:
            side = (f.get("side") or "").lower()
            qty = float(f.get("qty") or 0.0)
            px = float(f.get("price") or 0.0)
            fee = float(f.get("fee") or f.get("fees") or 0.0)
            fees_total += fee
            if qty <= 0 or px <= 0:
                continue
            if method.upper() == "WA":
                if side == "buy":
                    wa_cost = (wa_cost * wa_qty + px * qty) / (wa_qty + qty) if (wa_qty + qty) > 0 else px
                    wa_qty += qty
                elif side == "sell":
                    take = min(wa_qty, qty)
                    realized += (px - wa_cost) * take
                    wa_qty = max(0.0, wa_qty - take)
                    realized -= fee
            else:
                if side == "buy":
                    inv.append((qty, px))
                elif side == "sell":
                    remain = qty
                    while remain > 1e-12 and inv:
                        q0, p0 = inv[0]
                        take = min(q0, remain)
                        realized += (px - p0) * take
                        q0 -= take
                        remain -= take
                        if q0 <= 1e-12:
                            inv.popleft()
                        else:
                            inv[0] = (q0, p0)
                    realized -= fee
        # Inventario final
        if method.upper() == "WA":
            qty_open = float(wa_qty)
            avg_cost = float(wa_cost if wa_qty > 0 else 0.0)
        else:
            qty_open = float(sum(q for q, _ in inv))
            # promedio ponderado del inventario remanente
            denom = float(sum(q for q, _ in inv)) or 0.0
            avg_cost = float(sum(q * p for q, p in inv) / denom) if denom > 0 else 0.0
        last_price = float(_get_last_price(sym, db) or 0.0)
        unrealized = float((last_price - avg_cost) * qty_open) if qty_open > 0 and last_price > 0 else 0.0
        out.append({
            "symbol": sym,
            "qty_open": qty_open,
            "avg_cost": float(avg_cost),
            "last_price": last_price,
            "unrealized": float(unrealized),
            "realized_total": float(realized),
            "fees_total": float(fees_total),
        })
    return out

