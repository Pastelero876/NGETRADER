from __future__ import annotations

import time
import hmac
import hashlib
import math
import random
from typing import Any, Dict, List, Optional, Tuple

import requests


class BinanceBroker:
    """Adaptador de trading para Binance (spot) con validación de filtros y reintentos."""

    def __init__(self, api_key: str, api_secret: str, base_url: Optional[str] = None, testnet: bool = False) -> None:
        self.api_key = api_key
        self.api_secret = api_secret.encode("utf-8")
        self.base_url = base_url or ("https://testnet.binance.vision" if testnet else "https://api.binance.com")
        self._session = requests.Session()
        self._session.headers.update({"X-MBX-APIKEY": self.api_key})

    def _sign_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        query = "&".join(f"{k}={params[k]}" for k in sorted(params))
        signature = hmac.new(self.api_secret, query.encode("utf-8"), hashlib.sha256).hexdigest()
        params["signature"] = signature
        return params

    def _request_with_retry(self, method: str, path: str, params: Optional[Dict[str, Any]] = None, signed: bool = False) -> Any:
        url = f"{self.base_url}{path}"
        p = dict(params or {})
        if signed:
            p["timestamp"] = int(time.time() * 1000)
            p = self._sign_params(p)
        last_exc: Optional[Exception] = None
        for attempt in range(5):
            try:
                if method == "GET":
                    resp = self._session.get(url, params=p, timeout=30)
                elif method == "POST":
                    resp = self._session.post(url, params=p, timeout=30)
                elif method == "DELETE":
                    resp = self._session.delete(url, params=p, timeout=30)
                else:
                    raise ValueError("Método HTTP no soportado")
                resp.raise_for_status()
                return resp.json() if resp.content else {}
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                time.sleep(0.5 * (2 ** attempt) + random.random() * 0.2)
        if last_exc:
            raise last_exc
        return {}

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None, signed: bool = False) -> Any:
        return self._request_with_retry("GET", path, params=params, signed=signed)

    def _post(self, path: str, params: Optional[Dict[str, Any]] = None, signed: bool = False) -> Any:
        return self._request_with_retry("POST", path, params=params, signed=signed)

    def _delete(self, path: str, params: Optional[Dict[str, Any]] = None, signed: bool = False) -> Any:
        return self._request_with_retry("DELETE", path, params=params, signed=signed)

    # Utilidades de filtros/precisión
    def _get_symbol_filters(self, symbol: str) -> Dict[str, Dict[str, Any]]:
        info = self._get("/api/v3/exchangeInfo", params={"symbol": symbol.upper()})
        filters_list = info["symbols"][0]["filters"]
        return {f["filterType"]: f for f in filters_list}

    @staticmethod
    def _round_step(value: float, step: float) -> float:
        if step <= 0:
            return value
        return math.floor(value / step) * step

    def _apply_symbol_rules(self, symbol: str, quantity: Optional[float], price: Optional[float]) -> Tuple[Optional[float], Optional[float]]:
        filters = self._get_symbol_filters(symbol)
        # LOT_SIZE
        if quantity is not None and "LOT_SIZE" in filters:
            step = float(filters["LOT_SIZE"]["stepSize"])
            min_qty = float(filters["LOT_SIZE"]["minQty"])
            quantity = max(self._round_step(float(quantity), step), min_qty)
        # PRICE_FILTER
        if price is not None and "PRICE_FILTER" in filters:
            tick = float(filters["PRICE_FILTER"]["tickSize"])
            min_price = float(filters["PRICE_FILTER"].get("minPrice", 0) or 0)
            price = max(self._round_step(float(price), tick), min_price)
        return quantity, price

    # Trading API
    def place_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        type_: str = "MARKET",
        tif: str = "GTC",
        price: Optional[float] = None,
        post_only: bool = False,
        client_order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        order_type = type_.upper()
        if post_only and order_type in ("LIMIT", "MARKET"):
            # En spot, post-only se logra con LIMIT_MAKER
            order_type = "LIMIT_MAKER"
        adj_qty, adj_price = self._apply_symbol_rules(symbol, quantity, price)
        params: Dict[str, Any] = {
            "symbol": symbol.upper(),
            "side": side.upper(),
            "type": order_type,
        }
        if order_type in ("LIMIT", "LIMIT_MAKER"):
            if adj_price is None:
                raise ValueError("Se requiere price para órdenes LIMIT")
            params.update({"quantity": adj_qty, "price": adj_price, "timeInForce": tif.upper()})
        elif order_type == "MARKET":
            params.update({"quantity": adj_qty})
        else:
            params.update({"quantity": adj_qty})
            if adj_price is not None:
                params["price"] = adj_price
        if client_order_id:
            params["newClientOrderId"] = client_order_id
        return self._post("/api/v3/order", params=params, signed=True)

    def place_oco_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        stop_price: float,
        stop_limit_price: Optional[float] = None,
        tif: str = "GTC",
    ) -> Dict[str, Any]:
        adj_qty, adj_price = self._apply_symbol_rules(symbol, quantity, price)
        _, adj_stop = self._apply_symbol_rules(symbol, None, stop_price)
        _, adj_stop_limit = self._apply_symbol_rules(symbol, None, stop_limit_price if stop_limit_price is not None else stop_price)
        params: Dict[str, Any] = {
            "symbol": symbol.upper(),
            "side": side.upper(),
            "quantity": adj_qty,
            "price": adj_price,
            "stopPrice": adj_stop,
            "timeInForce": tif.upper(),
        }
        if adj_stop_limit is not None:
            params["stopLimitPrice"] = adj_stop_limit
        return self._post("/api/v3/order/oco", params=params, signed=True)

    def place_bracket_order(self, *args, **kwargs) -> Dict[str, Any]:  # noqa: ANN002, ANN003
        raise NotImplementedError("Binance spot no soporta 'bracket' estándar; usa OCO / lógica externa")

    def cancel_order(self, order_id: str, symbol: Optional[str] = None) -> Dict[str, Any]:
        if not symbol:
            raise ValueError("Se requiere symbol para cancelar en Binance")
        return self._delete("/api/v3/order", params={"symbol": symbol.upper(), "orderId": order_id}, signed=True)

    def cancel_all_orders(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        if not symbol:
            return {"status": "noop", "detail": "Binance requiere símbolo para cancelar todas las órdenes"}
        return self._delete("/api/v3/openOrders", params={"symbol": symbol.upper()}, signed=True)

    def list_orders(self, status: str = "open", limit: int = 100, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        p: Dict[str, Any] = {"limit": limit}
        if symbol:
            p["symbol"] = symbol.upper()
        if status == "open":
            return list(self._get("/api/v3/openOrders", params=p, signed=True))
        return list(self._get("/api/v3/allOrders", params=p, signed=True))

    def list_fills(self, symbol: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        if not symbol:
            return []
        data = list(self._get("/api/v3/myTrades", params={"symbol": symbol.upper(), "limit": limit}, signed=True))
        # Mapear comisión a 'fee' estándar
        for d in data:
            try:
                if "commission" in d and "fee" not in d:
                    d["fee"] = float(d.get("commission") or 0.0)
            except Exception:
                pass
        return data

    def get_account(self) -> Dict[str, Any]:
        return self._get("/api/v3/account", signed=True)

    def connectivity_ok(self) -> bool:
        try:
            _ = self._get("/api/v3/ping")
            return True
        except Exception:  # noqa: BLE001
            return False

