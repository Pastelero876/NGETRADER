from __future__ import annotations

import os
import time
from typing import Any

from nge_trader.adapters.binance_market_ws import BinanceMarketWS


def main() -> None:
    symbol = os.getenv("HARNESS_SYMBOL", "btcusdt").upper()
    depth = int(os.getenv("HARNESS_DEPTH", "10"))
    stream = os.getenv("HARNESS_STREAM", "depth").lower()
    testnet = bool(os.getenv("HARNESS_TESTNET", "1") == "1")
    ws = BinanceMarketWS(symbol=symbol, testnet=testnet, stream=stream, depth_levels=depth)

    def cb(evt: dict[str, Any]) -> None:
        print(evt)

    ws.start(cb)
    try:
        for _ in range(50):
            time.sleep(0.5)
    finally:
        ws.stop()


if __name__ == "__main__":
    main()


