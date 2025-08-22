from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Optional

import pandas as pd

from nge_trader.services.data_agg import stable_agg_trades


class LowLatencyProvider:
    """Proveedor de datos low-latency en memoria para flujos de trades.

    - Permite `push_event(symbol, price, qty, sequence_id, ts)` desde WS
    - Expone `get_vwap(symbol)` y `get_stable_agg(symbol)` basados en stable_agg_trades
    """

    def __init__(self, max_events: int = 5000) -> None:
        self.max_events = int(max_events)
        self._buf: Dict[str, Deque[dict]] = {}
        self._quotes: Dict[str, dict] = {}
        self._depth: Dict[str, dict] = {}

    def push_event(self, symbol: str, price: float, qty: float, sequence_id: Optional[int] = None, ts: Optional[float] = None) -> None:
        sym = symbol.upper()
        if sym not in self._buf:
            self._buf[sym] = deque(maxlen=self.max_events)
        self._buf[sym].append({
            "sequence_id": int(sequence_id) if sequence_id is not None else None,
            "ts": float(ts) if ts is not None else None,
            "price": float(price),
            "qty": float(qty),
        })

    def get_stable_agg(self, symbol: str) -> pd.DataFrame:
        sym = symbol.upper()
        buf = list(self._buf.get(sym, []))
        if not buf:
            return pd.DataFrame(columns=["sequence_id", "ts", "price", "qty", "vwap20", "vol20"])
        df = pd.DataFrame(buf)
        return stable_agg_trades(df)

    def get_vwap(self, symbol: str) -> Optional[float]:
        df = self.get_stable_agg(symbol)
        if df.empty:
            return None
        try:
            return float(df.iloc[-1]["vwap20"]) if "vwap20" in df.columns else None
        except Exception:
            return None

    def get_volatility(self, symbol: str, window: int = 20) -> Optional[float]:
        df = self.get_stable_agg(symbol)
        if df.empty:
            return None
        try:
            px = df["price"].astype(float).tail(window)
            rets = px.pct_change().dropna()
            return float(rets.std(ddof=0)) if not rets.empty else 0.0
        except Exception:
            return None

    # ====== L1 Quotes (bid/ask) ======
    def push_quote(
        self,
        symbol: str,
        bid: float,
        ask: float,
        bid_size: float | None = None,
        ask_size: float | None = None,
        ts: float | None = None,
    ) -> None:
        sym = symbol.upper()
        self._quotes[sym] = {
            "bid": float(bid),
            "ask": float(ask),
            "bid_size": float(bid_size) if bid_size is not None else None,
            "ask_size": float(ask_size) if ask_size is not None else None,
            "ts": float(ts) if ts is not None else None,
        }

    # ====== L2 Depth (top N niveles) ======
    def push_depth(self, symbol: str, bids: list[tuple[float, float]] | list[list[float]] | None, asks: list[tuple[float, float]] | list[list[float]] | None, ts: float | None = None) -> None:
        sym = symbol.upper()
        def _norm(arr):
            out: list[tuple[float, float]] = []
            for it in list(arr or []):
                try:
                    p, q = (float(it[0]), float(it[1]))
                    out.append((p, q))
                except Exception:
                    continue
            return out
        self._depth[sym] = {
            "bids": _norm(bids or []),
            "asks": _norm(asks or []),
            "ts": float(ts) if ts is not None else None,
        }

    def get_depth(self, symbol: str) -> dict:
        return dict(self._depth.get(symbol.upper(), {"bids": [], "asks": [], "ts": None}))

    def get_l2_ofi(self, symbol: str, levels: int = 5) -> float:
        d = self.get_depth(symbol)
        bids = list(d.get("bids") or [])[:levels]
        asks = list(d.get("asks") or [])[:levels]
        vb = float(sum(q for _, q in bids))
        va = float(sum(q for _, q in asks))
        if (vb + va) <= 0:
            return 0.0
        return float((vb - va) / (vb + va))

    def get_l2_microprice(self, symbol: str, levels: int = 5) -> float | None:
        d = self.get_depth(symbol)
        bids = list(d.get("bids") or [])[:levels]
        asks = list(d.get("asks") or [])[:levels]
        if not bids or not asks:
            return None
        pb, qb = bids[0]
        pa, qa = asks[0]
        if pb <= 0 or pa <= 0:
            return None
        if qb > 0 and qa > 0:
            return float((pa * qb + pb * qa) / (qb + qa))
        return float((pa + pb) / 2.0)

    def get_microprice(self, symbol: str) -> Optional[float]:
        sym = symbol.upper()
        q = self._quotes.get(sym)
        if not q:
            return None
        b = float(q.get("bid") or 0.0)
        a = float(q.get("ask") or 0.0)
        bs = float(q.get("bid_size") or 0.0)
        asz = float(q.get("ask_size") or 0.0)
        if b <= 0 or a <= 0:
            return None
        if bs > 0 and asz > 0:
            # Microprice ponderada por tamaños
            return float((a * bs + b * asz) / (bs + asz))
        # Sin tamaños, punto medio
        return float((a + b) / 2.0)

    def get_quote_metrics(self, symbol: str) -> dict:
        sym = symbol.upper()
        q = self._quotes.get(sym) or {}
        b = float(q.get("bid") or 0.0)
        a = float(q.get("ask") or 0.0)
        bs = float(q.get("bid_size") or 0.0)
        asz = float(q.get("ask_size") or 0.0)
        spread = float(a - b) if (a > 0 and b > 0) else 0.0
        mp = self.get_microprice(sym) or 0.0
        ofi = 0.0
        if (bs + asz) > 0:
            ofi = float((bs - asz) / (bs + asz))
        obi = 0.0
        if bs > 0 and asz > 0:
            obi = float(bs - asz)
        return {
            "bid": b,
            "ask": a,
            "bid_size": bs,
            "ask_size": asz,
            "spread": spread,
            "microprice": float(mp),
            "ofi": ofi,
            "obi": obi,
            "ts": q.get("ts"),
        }

    def get_ofi_obi(self, symbol: str) -> tuple[float, float]:
        m = self.get_quote_metrics(symbol)
        return float(m.get("ofi") or 0.0), float(m.get("obi") or 0.0)

    def suggest_limit_price(self, symbol: str, side: str, base_price: float, bps: float = 0.0005) -> float:
        """Sugiere un precio límite ajustado por microprice/OFI.

        - Si hay microprice, usarlo como referencia; si no, usar base_price
        - Ajustar delta por bps y por OFI (más agresivo si OFI favorece nuestro lado)
        """
        q = self.get_quote_metrics(symbol)
        mp = float(q.get("microprice") or 0.0) or float(base_price)
        ofi = float(q.get("ofi") or 0.0)
        # multiplicador por ofi en rango [0.5, 1.5]
        mult = max(0.5, min(1.5, 1.0 + ofi))
        delta = float(mp) * float(bps) * mult
        if str(side).lower() == "buy":
            return float(mp - delta)
        return float(mp + delta)

    # ====== VPIN (bulk volume classification aproximado) ======
    def get_vpin(self, symbol: str, target_buckets: int = 50) -> Optional[float]:
        """Calcula VPIN simple usando dirección por tick y buckets por volumen.

        - Se acumulan trades en buckets de volumen uniforme (total_vol/target_buckets)
        - En cada bucket se calcula |vol_compra - vol_venta| / (vol_compra + vol_venta)
        - VPIN es el promedio de los últimos buckets disponibles
        """
        sym = symbol.upper()
        events = list(self._buf.get(sym, []))
        if len(events) < 5:
            return None
        total_vol = float(sum(float(e.get("qty") or 0.0) for e in events))
        if total_vol <= 0:
            return None
        bucket_vol = max(total_vol / max(1, int(target_buckets)), 1e-9)
        last_price = None
        cur_buy = 0.0
        cur_sell = 0.0
        buckets: List[float] = []
        acc_vol = 0.0
        for e in events:
            qty = float(e.get("qty") or 0.0)
            price = float(e.get("price") or 0.0)
            if last_price is None:
                last_price = price
            sign = 1.0 if price >= last_price else -1.0
            buy_part = qty if sign > 0 else 0.0
            sell_part = qty if sign < 0 else 0.0
            cur_buy += buy_part
            cur_sell += sell_part
            acc_vol += qty
            last_price = price
            while acc_vol >= bucket_vol:
                # cerrar bucket
                denom = max(cur_buy + cur_sell, 1e-9)
                buckets.append(abs(cur_buy - cur_sell) / denom)
                acc_vol -= bucket_vol
                # arrastrar remanente proporcional (sencillo: reset bucket)
                cur_buy = 0.0
                cur_sell = 0.0
        if not buckets:
            return None
        return float(sum(buckets[-target_buckets:]) / len(buckets[-target_buckets:]))


