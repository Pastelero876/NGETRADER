from __future__ import annotations

from typing import Literal, Any, Dict, List
import time
import uuid


class PaperBroker:
    """Broker simulado (paper trading)."""
    def __init__(self) -> None:
        self._orders: Dict[str, Dict[str, Any]] = {}
        self._fills: List[Dict[str, Any]] = []
        self._positions: Dict[str, Dict[str, Any]] = {}
        self._equity: float = 100_000.0

    def place_order(
        self, symbol: str, side: Literal["buy", "sell"], quantity: float, price: float | None = None, **kwargs: Any
    ) -> dict:
        """Acepta órdenes sin ejecutar en mercado real."""
        # idempotencia por client_order_id
        client_oid = (kwargs or {}).get("client_order_id")
        if client_oid:
            for o in self._orders.values():
                if o.get("client_order_id") == client_oid:
                    return o
        oid = str(uuid.uuid4())
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        order = {
            "id": oid,
            "status": "accepted",
            "mode": "paper",
            "symbol": symbol,
            "side": side,
            "qty": float(quantity),
            "price": price,
            "created_at": ts,
            "order_class": None,
            "client_order_id": client_oid,
        }
        self._orders[oid] = order
        return order

    def place_bracket_order(
        self,
        symbol: str,
        side: Literal["buy", "sell"],
        quantity: float,
        take_profit_price: float | None = None,
        stop_loss_price: float | None = None,
        type_: str = "market",
        tif: str = "gtc",
    ) -> Dict[str, Any]:
        base = self.place_order(symbol, side, quantity)
        base["order_class"] = "bracket"
        base["take_profit"] = {"limit_price": take_profit_price} if take_profit_price else None
        base["stop_loss"] = {"stop_price": stop_loss_price} if stop_loss_price else None
        base["type"] = type_
        base["time_in_force"] = tif
        self._orders[base["id"]] = base
        return base

    def list_positions(self) -> list[dict]:
        """Devuelve posiciones abiertas (vacío en modo paper básico)."""
        return list(self._positions.values())

    def list_orders(self, status: str = "open", limit: int = 100) -> List[Dict[str, Any]]:
        out = list(self._orders.values())
        if status and status != "all":
            out = [o for o in out if (o.get("status") or "").lower() in (status, "accepted", "open")]
        return out[:limit]

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        o = self._orders.get(order_id)
        if not o:
            return {"status": "not_found", "order_id": order_id}
        o["status"] = "canceled"
        return {"status": "canceled", "order_id": order_id}

    def cancel_all_orders(self, symbol: str | None = None) -> Dict[str, Any]:
        canceled = 0
        for o in self._orders.values():
            if symbol and (o.get("symbol") or "").upper() != symbol.upper():
                continue
            if (o.get("status") or "").lower() in ("accepted", "open"):
                o["status"] = "canceled"
                canceled += 1
        return {"status": "ok", "canceled": canceled}

    def replace_order(self, order_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        o = self._orders.get(order_id)
        if not o:
            return {"status": "not_found", "order_id": order_id}
        o.update(payload)
        o["status"] = "replaced"
        return {"status": "replaced", "order_id": order_id, "payload": payload}

    def list_fills(self) -> List[Dict[str, Any]]:
        return list(self._fills)

    def get_account(self) -> Dict[str, Any]:
        return {
            "equity": f"{self._equity:.2f}",
            "last_equity": f"{self._equity:.2f}",
            "unrealized_pl": 0.0,
        }

    def close_all_positions(self) -> Dict[str, Any]:
        self._positions.clear()
        return {"status": "ok"}

    def close_symbol_position(self, symbol: str) -> Dict[str, Any]:
        self._positions.pop(symbol, None)
        return {"status": "ok", "symbol": symbol}


