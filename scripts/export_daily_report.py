from __future__ import annotations

import csv
from pathlib import Path
from datetime import datetime, UTC

from nge_trader.repository.db import Database
from nge_trader.services.accounting import compute_pnl_summary


def export_daily_report(out_dir: str = "reports") -> str:
    db = Database()
    ts = datetime.now(UTC).strftime("%Y-%m-%d")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Fills del día
    fills = [r for r in db.recent_fills(10000) if r.get("ts", "").startswith(ts)]
    fills_path = Path(out_dir) / f"fills_{ts}.csv"
    with open(fills_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["ts", "symbol", "side", "qty", "price", "fees", "order_id", "liquidity"]) 
        w.writeheader()
        for r in fills:
            w.writerow({k: r.get(k) for k in w.fieldnames})

    # Órdenes del día
    orders = [r for r in db.recent_orders(10000) if r.get("ts", "").startswith(ts)]
    orders_path = Path(out_dir) / f"orders_{ts}.csv"
    with open(orders_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["ts", "symbol", "side", "qty", "price", "status", "order_id"]) 
        w.writeheader()
        for r in orders:
            w.writerow({k: r.get(k) for k in w.fieldnames})

    # Resumen P&L por símbolo (trades cerrados hoy)
    trades = [r for r in db.recent_trades(10000) if str(r.get("out_time") or "").startswith(ts)]
    pnl_by_symbol: dict[str, dict[str, float]] = {}
    for t in trades:
        sym = str(t.get("symbol") or "")
        pnl_by_symbol.setdefault(sym, {"realized": 0.0, "fees": 0.0})
        pnl_by_symbol[sym]["realized"] += float(t.get("realized") or 0.0)
        pnl_by_symbol[sym]["fees"] += float(t.get("fees") or 0.0)
    pnl_path = Path(out_dir) / f"pnl_{ts}.csv"
    with open(pnl_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["symbol", "realized", "fees"]) 
        w.writeheader()
        for sym, agg in sorted(pnl_by_symbol.items()):
            w.writerow({"symbol": sym, "realized": agg["realized"], "fees": agg["fees"]})

    # Snapshot de P&L no realizado (inventario actual)
    unreal = compute_pnl_summary()
    unreal_path = Path(out_dir) / f"unrealized_{ts}.csv"
    with open(unreal_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["symbol", "qty_open", "avg_cost", "last_price", "unrealized", "realized_total", "fees_total"]) 
        w.writeheader()
        for row in unreal:
            w.writerow(row)

    return str(Path(out_dir).resolve())


if __name__ == "__main__":
    out = export_daily_report()
    print(out)

