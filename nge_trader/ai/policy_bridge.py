from __future__ import annotations

from nge_trader.common.contracts import TradeIntent


def decide(state, edge_bps: float) -> TradeIntent:
    """Decisión de política mínima para preview.

    Espera que `state` tenga atributos/campos: symbol, costs.fees_bps, costs.half_spread_bps.
    """
    try:
        fees = float(getattr(getattr(state, "costs", object()), "fees_bps", 0.0))
        hs = float(getattr(getattr(state, "costs", object()), "half_spread_bps", 0.0))
    except Exception:
        fees = 0.0
        hs = 0.0
    if float(edge_bps) < (fees + hs + 2.0):
        return TradeIntent(getattr(state, "symbol", "?"), "no-trade", 0.0, 0, float(edge_bps), "edge<cost")
    return TradeIntent(getattr(state, "symbol", "?"), "buy", 0.6, 240, float(edge_bps), "regime=trend")


