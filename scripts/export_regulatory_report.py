from __future__ import annotations

import csv
from pathlib import Path

from nge_trader.repository.db import Database


def export_orders_csv(out_path: str = "reports/regulatory_orders.csv") -> str:
    db = Database()
    rows = db.recent_orders(10000)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "ts",
        "symbol",
        "side",
        "qty",
        "price",
        "status",
        "order_id",
        "decision_id",
        "execution_id",
        "dea",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows[::-1]:
            w.writerow({k: r.get(k) for k in fields})
    return out_path


if __name__ == "__main__":
    path = export_orders_csv()
    print(f"Exported: {path}")


