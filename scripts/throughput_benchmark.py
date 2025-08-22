from __future__ import annotations

"""Benchmark sencillo de throughput y latencia (paper).

Uso:
  BROKER=paper python scripts/throughput_benchmark.py --symbol BTCUSDT --n 200
"""

import argparse
import time

from nge_trader.services.app_service import AppService
from nge_trader.services.oms import place_market_order
from nge_trader.services.metrics import observe_latency


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--n", type=int, default=100)
    args = parser.parse_args()

    svc = AppService()
    latencies: list[float] = []
    t_start = time.time()
    for _ in range(int(args.n)):
        t0 = time.time()
        try:
            place_market_order(svc.broker, args.symbol, "buy", 0.001)
        except Exception:
            pass
        dt_ms = (time.time() - t0) * 1000.0
        latencies.append(dt_ms)
        observe_latency("order_place_latency_ms", dt_ms)
    t_end = time.time()
    elapsed = t_end - t_start
    throughput = float(args.n) / max(elapsed, 1e-9)
    lat_sorted = sorted(latencies) or [0.0]
    p50 = lat_sorted[int(0.5 * (len(lat_sorted) - 1))]
    p95 = lat_sorted[int(0.95 * (len(lat_sorted) - 1))]
    print({
        "orders": int(args.n),
        "elapsed_sec": round(elapsed, 4),
        "throughput_ops_per_sec": round(throughput, 2),
        "p50_ms": round(p50, 2),
        "p95_ms": round(p95, 2),
    })


if __name__ == "__main__":
    main()


