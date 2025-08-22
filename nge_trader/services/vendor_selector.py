from __future__ import annotations

from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class SymbolCost:
    symbol: str
    liquidity: float
    spread_pct: float
    fee_pct: float
    min_notional: float
    tradability: float


DEFAULT_VENDOR_STATS: Dict[str, Dict[str, float]] = {
    # Data providers
    "ALPHA_VANTAGE": {"sla": 0.98, "uptime": 0.995, "trust": 0.8, "support": 0.6},
    "BINANCE_DATA": {"sla": 0.99, "uptime": 0.999, "trust": 0.85, "support": 0.6},
    "TIINGO": {"sla": 0.995, "uptime": 0.999, "trust": 0.9, "support": 0.8},
    "POLYGON": {"sla": 0.995, "uptime": 0.999, "trust": 0.92, "support": 0.8},
    "KAIKO": {"sla": 0.995, "uptime": 0.999, "trust": 0.93, "support": 0.75},
    "ALPACA_DATA": {"sla": 0.99, "uptime": 0.998, "trust": 0.88, "support": 0.8},
    # Brokers
    "PAPER": {"sla": 1.0, "uptime": 1.0, "trust": 0.5, "support": 0.5},
    "ALPACA_BROKER": {"sla": 0.995, "uptime": 0.999, "trust": 0.9, "support": 0.85},
    "BINANCE_BROKER": {"sla": 0.995, "uptime": 0.999, "trust": 0.88, "support": 0.7},
    "COINBASE_BROKER": {"sla": 0.995, "uptime": 0.999, "trust": 0.92, "support": 0.8},
    "IBKR_BROKER": {"sla": 0.999, "uptime": 0.999, "trust": 0.95, "support": 0.9},
    "KRAKEN_BROKER": {"sla": 0.995, "uptime": 0.998, "trust": 0.9, "support": 0.75},
}


MARKET_TO_DATA_PROVIDERS: Dict[str, List[str]] = {
    "EQUITIES": ["TIINGO", "POLYGON", "ALPACA_DATA"],
    "CRYPTO": ["BINANCE_DATA", "KAIKO", "ALPHA_VANTAGE"],
    "FX": ["ALPHA_VANTAGE"],
}

MARKET_TO_BROKERS: Dict[str, List[str]] = {
    "EQUITIES": ["IBKR_BROKER", "ALPACA_BROKER"],
    "CRYPTO": ["BINANCE_BROKER", "COINBASE_BROKER", "KRAKEN_BROKER"],
}


def compute_vendor_score(sla: float, uptime: float, trust: float, support: float, weights: Tuple[float, float, float, float] = (0.35, 0.25, 0.25, 0.15)) -> float:
    w_sla, w_up, w_tr, w_sup = weights
    return float(
        w_sla * float(sla) +
        w_up * float(uptime) +
        w_tr * float(trust) +
        w_sup * float(support)
    )


def recommend_vendors(kind: str, market: str, region: str | None = None, top_k: int = 3) -> List[Dict[str, str | float]]:
    """Return a ranked list of vendors for given market.

    kind: "data" | "broker"
    market: "EQUITIES" | "CRYPTO" | "FX" | ...
    region: optional region filter (currently advisory only)
    """
    mk = (market or "").strip().upper() or "CRYPTO"
    kd = (kind or "data").strip().lower()
    if kd not in {"data", "broker"}:
        kd = "data"
    candidates = (MARKET_TO_DATA_PROVIDERS if kd == "data" else MARKET_TO_BROKERS).get(mk, [])
    out: List[Dict[str, str | float]] = []
    for name in candidates:
        stats = DEFAULT_VENDOR_STATS.get(name, {})
        score = compute_vendor_score(stats.get("sla", 0.0), stats.get("uptime", 0.0), stats.get("trust", 0.0), stats.get("support", 0.0))
        reason = f"sla={stats.get('sla', 0):.3f}, uptime={stats.get('uptime', 0):.3f}, trust={stats.get('trust', 0):.2f}, support={stats.get('support', 0):.2f}"
        if region:
            reason += f", region={region}"
        out.append({"name": name, "score": float(score), "reason": reason})
    out = sorted(out, key=lambda r: float(r.get("score", 0.0)), reverse=True)[: max(int(top_k), 1)]
    return out


def score_symbol_tradability(symbol: str, liquidity: float, spread_pct: float, fee_pct: float, min_notional: float, expected_edge_bps: float, max_spread_edge_ratio: float = 0.5) -> SymbolCost | None:
    """Calcula score de tradability y excluye si notional/spread comen el edge.

    tradability = liquidez - spread% - fee% - min_notional_penalty
    """
    try:
        penalty = 0.0
        edge = float(expected_edge_bps)
        # Si half-spread en bps supera ratio*edge esperado, penalización alta
        if (spread_pct * 10000.0) > float(max_spread_edge_ratio) * edge:
            penalty += spread_pct
        trad = float(liquidity) - float(spread_pct) - float(fee_pct) - penalty
        return SymbolCost(symbol=symbol, liquidity=float(liquidity), spread_pct=float(spread_pct), fee_pct=float(fee_pct), min_notional=float(min_notional), tradability=float(trad))
    except Exception:
        return None


def select_universe_by_cost(candidates: List[Dict[str, float]], expected_edge_bps: float, max_spread_edge_ratio: float = 0.5, top_k: int = 10) -> List[SymbolCost]:
    """Selecciona símbolos más operables por coste/edge.

    candidates: [{'symbol','liquidity','spread_pct','fee_pct','min_notional'}]
    """
    scored: List[SymbolCost] = []
    for c in candidates:
        sc = score_symbol_tradability(
            symbol=str(c.get('symbol')),
            liquidity=float(c.get('liquidity') or 0.0),
            spread_pct=float(c.get('spread_pct') or 0.0),
            fee_pct=float(c.get('fee_pct') or 0.0),
            min_notional=float(c.get('min_notional') or 0.0),
            expected_edge_bps=float(expected_edge_bps),
            max_spread_edge_ratio=float(max_spread_edge_ratio),
        )
        if sc is not None:
            scored.append(sc)
    scored = [s for s in scored if (s.spread_pct * 10000.0) <= max_spread_edge_ratio * expected_edge_bps]
    scored = sorted(scored, key=lambda s: s.tradability, reverse=True)
    return scored[: max(int(top_k), 1)]


