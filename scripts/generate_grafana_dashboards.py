from __future__ import annotations

import json


def main() -> None:
    # Stub: genera un JSON de panel mínimo
    dashboard = {
        "title": "NGEtrader Live",
        "panels": [
            {
                "type": "graph",
                "title": "Order Latency (ms) p50/p95",
                "targets": [
                    {"expr": "order_place_latency_ms_p50"},
                    {"expr": "order_place_latency_ms_p95"},
                ],
            },
            {"type": "graph", "title": "Slippage bps", "targets": [{"expr": "slippage_bps"}]},
            {"type": "graph", "title": "Fill Ratio", "targets": [{"expr": "fill_ratio"}]},
            {"type": "graph", "title": "WS Uptime (s)", "targets": [{"expr": "ws_uptime_seconds"}]},
            {"type": "stat", "title": "WS Reconnects", "targets": [{"expr": "ws_reconnects_total"}]},
            {"type": "stat", "title": "Error Rate Recent", "targets": [{"expr": "slo_error_rate_recent"}]},
            {"type": "stat", "title": "Rate Limit Remaining (broker)", "targets": [{"expr": "rate_limit_broker_remaining"}]},
            {"type": "stat", "title": "Rate Limit Remaining (strategy)", "targets": [{"expr": "rate_limit_strategy_remaining"}]},
            {"type": "stat", "title": "Breaker Activations", "targets": [{"expr": "breaker_activations_total"}]},
            {"type": "graph", "title": "VPIN", "targets": [{"expr": "vpin"}]},
            {"type": "graph", "title": "L2 OFI", "targets": [{"expr": "l2_ofi"}]},
            {"type": "graph", "title": "L2 microprice", "targets": [{"expr": "l2_microprice"}]},
        ],
        "annotations": {
            "list": [
                {"name": "High Error Rate", "expr": "slo_error_rate_recent > 0.2"},
                {"name": "High Latency", "expr": "order_place_latency_ms_p95 > 300"},
                {"name": "Drift PSI Alto", "expr": "model_drift_psi > 0.2"},
                {"name": "Slippage Alto", "expr": "slippage_bps_p95 > 25"},
                {"name": "OFI Alto", "expr": "l2_ofi > 0.5"},
            ]
        },
    }
    print(json.dumps(dashboard))


if __name__ == "__main__":
    main()


