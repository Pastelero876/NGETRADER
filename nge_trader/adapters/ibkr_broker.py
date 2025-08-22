from __future__ import annotations

from typing import Any, Dict, List, Optional


class IBKRBroker:
    """Adaptador mínimo para Interactive Brokers usando ib_insync (opcional).

    Si ib_insync no está instalado o no se puede conectar al TWS/Gateway, las
    operaciones levantarán excepción para que el envoltorio de resiliencia haga
    failover a paper.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 7497, client_id: int = 1) -> None:
        try:
            from ib_insync import IB  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("ib_insync no está instalado") from exc
        self._IB = IB  # guardar clase para references
        self.ib = IB()
        if not self.ib.isConnected():
            self.ib.connect(host, int(port), clientId=int(client_id), readonly=False, timeout=10)

    def _contract(self, symbol: str, sec_type: str = "STK", exchange: str = "SMART", currency: str = "USD"):
        from ib_insync import Stock, Future  # type: ignore
        if sec_type.upper() == "STK":
            return Stock(symbol, exchange, currency)
        if sec_type.upper() == "FUT":
            # Futuro continuo genérico; para producción se debe especificar vencimiento
            return Future(symbol, exchange, currency)
        raise ValueError("sec_type no soportado")

    def get_account(self) -> Dict[str, Any]:
        # Equity aproximado via accountSummary
        acc = {"equity": None, "last_equity": None, "unrealized_pl": None}
        try:
            summary = {row.tag: row.value for row in self.ib.accountSummary()}
            equity = float(summary.get("NetLiquidation", 0))
            acc.update({"equity": equity, "last_equity": equity, "unrealized_pl": float(summary.get("UnrealizedPnL", 0))})
        except Exception:  # noqa: BLE001
            pass
        return acc

    def connectivity_ok(self) -> bool:
        try:
            return bool(self.ib.isConnected())
        except Exception:  # noqa: BLE001
            return False

    def place_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        type_: str = "MKT",
        price: Optional[float] = None,
        sec_type: str = "STK",
        exchange: str = "SMART",
        currency: str = "USD",
        client_order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        from ib_insync import MarketOrder, LimitOrder  # type: ignore
        contract = self._contract(symbol, sec_type, exchange, currency)
        action = side.upper()
        if type_.upper() == "MKT":
            order = MarketOrder(action, quantity)
        elif type_.upper() == "LMT":
            if price is None:
                raise ValueError("Se requiere price para LMT en IBKR")
            order = LimitOrder(action, quantity, float(price))
        else:
            raise ValueError("type_ no soportado en implementación mínima")
        if client_order_id:
            try:
                order.orderRef = str(client_order_id)
            except Exception:
                pass
        trade = self.ib.placeOrder(contract, order)
        self.ib.sleep(0)  # procesar eventos
        return {
            "orderId": getattr(trade, "order", None) and trade.order.orderId,
            "status": getattr(trade, "orderStatus", None) and trade.orderStatus.status,
            "symbol": symbol,
            "side": side,
            "qty": float(quantity),
            "type": type_.upper(),
            "price": price,
        }

    def cancel_order(self, order_id: int) -> Dict[str, Any]:
        # En ib_insync se necesita un objeto Order; tratamos de obtenerlo del openOrders
        for o in self.ib.openOrders():
            if o.orderId == int(order_id):
                self.ib.cancelOrder(o)
                return {"status": "canceled", "order_id": order_id}
        return {"status": "not_found", "order_id": order_id}

    def cancel_all_orders(self) -> Dict[str, Any]:
        canceled = 0
        for o in list(self.ib.openOrders()):
            try:
                self.ib.cancelOrder(o)
                canceled += 1
            except Exception:
                pass
        return {"status": "ok", "canceled": canceled}

    def cancel_all_orders_by_symbol(self, symbol: str) -> Dict[str, Any]:
        """Intenta cancelar órdenes abiertas cuyo contrato coincida con `symbol`."""
        canceled = 0
        try:
            symu = symbol.upper()
            # Abrir trades para tener acceso a contract
            for t in self.ib.trades():
                try:
                    if t.orderStatus.status.lower() in ("submitted", "presubmitted") and getattr(t.contract, "symbol", "").upper() == symu:
                        self.ib.cancelOrder(t.order)
                        canceled += 1
                except Exception:
                    continue
        except Exception:
            pass
        return {"status": "ok", "canceled": canceled}

    def list_orders(self, status: str = "open", limit: int = 100) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if status == "open":
            for o in self.ib.openOrders()[:limit]:
                out.append({"orderId": o.orderId, "status": "open", "symbol": getattr(o, "symbol", None)})
        else:
            for t in self.ib.trades()[:limit]:
                out.append({"orderId": t.order.orderId, "status": t.orderStatus.status, "symbol": getattr(t.contract, "symbol", None)})
        return out

    def list_positions(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for p in self.ib.positions():
            out.append({
                "symbol": p.contract.symbol,
                "qty": float(p.position),
                "avg_price": float(getattr(p, "avgCost", 0.0)),
            })
        return out

    def list_fills(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for t in self.ib.trades():
            try:
                filled_ok = bool(getattr(t.orderStatus, "filled", 0)) or (getattr(t.orderStatus, "status", "").lower() in ("filled", "partial", "partiallyfilled"))
            except Exception:  # noqa: BLE001
                filled_ok = False
            if filled_ok and getattr(t, "fills", None):
                for f in t.fills:
                    try:
                        exec_obj = getattr(f, "execution", None)
                        cr = getattr(f, "commissionReport", None)
                        fee_val = None
                        if cr is not None and hasattr(cr, "commission"):
                            fee_val = float(getattr(cr, "commission", 0.0) or 0.0)
                        elif exec_obj is not None and hasattr(exec_obj, "commission"):
                            fee_val = float(getattr(exec_obj, "commission", 0.0) or 0.0)
                        else:
                            fee_val = 0.0
                        out.append({
                            "order_id": getattr(t.order, "orderId", None),
                            "symbol": getattr(getattr(t, "contract", None), "symbol", None),
                            "qty": float(getattr(exec_obj, "shares", 0.0)) if exec_obj is not None else 0.0,
                            "price": float(getattr(exec_obj, "price", 0.0)) if exec_obj is not None else 0.0,
                            "ts": getattr(exec_obj, "time", None) if exec_obj is not None else None,
                            "side": getattr(t.order, "action", "").lower() if getattr(t, "order", None) else None,
                            "fee": float(fee_val or 0.0),
                        })
                    except Exception:
                        continue
        return out

    def place_bracket_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        take_profit_price: float,
        stop_loss_price: float,
        sec_type: str = "STK",
        exchange: str = "SMART",
        currency: str = "USD",
    ) -> Dict[str, Any]:
        contract = self._contract(symbol, sec_type, exchange, currency)
        # Intentar helper nativo de ib_insync
        bracket_fn = getattr(self.ib, "bracketOrder", None)
        if callable(bracket_fn):
            orders = bracket_fn(side.upper(), quantity, take_profit_price, stop_loss_price)
            for o in orders:
                self.ib.placeOrder(contract, o)
            self.ib.sleep(0)
            parent_id = getattr(orders[0], "orderId", None)
            return {"status": "submitted", "parent_order_id": parent_id, "count": len(orders)}
        raise NotImplementedError("Esta instalación de ib_insync no soporta bracketOrder helper")

    # ===== Reauth/Re-subscribe stubs =====
    def reconnect(self) -> bool:
        try:
            if not self.ib.isConnected():
                self.ib.connect("127.0.0.1", 7497, clientId=1, readonly=False, timeout=10)
            return bool(self.ib.isConnected())
        except Exception:
            return False

    def backfill_recent(self, limit: int = 50) -> Dict[str, Any]:
        """Recupera últimas órdenes y fills desde la sesión actual (no persiste)."""
        orders: List[Dict[str, Any]] = []
        try:
            for t in self.ib.trades()[:limit]:
                try:
                    orders.append({
                        "orderId": getattr(t.order, "orderId", None),
                        "status": getattr(t.orderStatus, "status", None),
                        "symbol": getattr(getattr(t, "contract", None), "symbol", None),
                    })
                except Exception:
                    continue
        except Exception:
            pass
        return {"orders": orders, "fills": self.list_fills()[:limit]}


