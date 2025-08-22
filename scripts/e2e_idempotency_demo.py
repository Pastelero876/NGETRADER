from __future__ import annotations

import os
import time

from nge_trader.services.app_service import AppService
from nge_trader.services.oms import place_market_order


def main() -> None:
    os.environ.setdefault("PROFILE", "dev")
    svc = AppService()
    symbol = os.environ.get("E2E_SYMBOL", "BTCUSDT")
    # Enviar dos veces consecutivas la misma orden; el broker debe ignorar duplicados por client id
    o1 = place_market_order(svc.broker, symbol, "buy", 0.001)
    time.sleep(0.2)
    o2 = place_market_order(svc.broker, symbol, "buy", 0.001)
    print("ORDER1:", o1)
    print("ORDER2:", o2)


if __name__ == "__main__":
    main()


