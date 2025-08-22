from __future__ import annotations

"""Ejemplo reproducible: colocar una orden de market en broker paper y mostrar summary.

Uso:
  DATA_PROVIDER=alpha_vantage BROKER=paper python scripts/example_paper_trade.py
"""

from nge_trader.services.app_service import AppService
from nge_trader.services.oms import place_market_order


def main() -> None:
    svc = AppService()
    sym = "AAPL"
    try:
        res = place_market_order(svc.broker, sym, "buy", 1.0)
        print({"order": res})
    except Exception as exc:  # noqa: BLE001
        print({"error": str(exc)})
    # Summary
    print({"connectivity": svc.get_connectivity_status()})


if __name__ == "__main__":
    main()


