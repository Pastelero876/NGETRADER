from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class ShadowEvent:
    symbol: str
    ts: float
    side: str
    est_price: float
    ref_px: float
    est_slippage_bps: float
    route_algo: str
    strategy: str
    qty: float
    meta: dict


def write(db, ev: ShadowEvent, table: str = "tca_shadow_events") -> None:
    from sqlalchemy import text  # type: ignore
    q = text(
        f"""
INSERT INTO {table}
(symbol, ts, side, est_price, ref_px, est_slippage_bps, route_algo, strategy, qty, meta)
VALUES (:symbol, to_timestamp(:ts), :side, :est_price, :ref_px, :est_slippage_bps, :route_algo, :strategy, :qty, CAST(:meta AS JSONB))
        """
    )
    payload = asdict(ev)
    db.execute(q, {**payload, "meta": ev.meta or {}})


