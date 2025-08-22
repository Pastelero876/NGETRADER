from __future__ import annotations

import base64
import hashlib
import hmac
import json as _json
import time
from typing import Any, Dict, List, Optional

import requests


class CoinbaseBroker:
    """Adaptador mínimo para Coinbase Exchange (spot) usando REST v2.

    Requiere API Key, API Secret (base64) y Passphrase.
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        passphrase: str,
        base_url: str | None = None,
    ) -> None:
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        self.base_url = base_url or "https://api.exchange.coinbase.com"
        self._session = requests.Session()

    def _sign(self, timestamp: str, method: str, request_path: str, body: str = "") -> str:
        key = base64.b64decode(self.api_secret)
        message = f"{timestamp}{method}{request_path}{body}".encode("utf-8")
        h = hmac.new(key, message, hashlib.sha256)
        return base64.b64encode(h.digest()).decode("utf-8")

    def _request(self, method: str, path: str, params: Optional[Dict[str, Any]] = None, json: Optional[Dict[str, Any]] = None) -> Any:  # noqa: A002
        url = f"{self.base_url}{path}"
        body_str = _json.dumps(json) if json else ""
        ts = str(int(time.time()))
        req_path = path
        if params:
            # Coinbase firma sin query en la cadena; pero la URL sí lleva query
            url = url + ("?" if "?" not in url else "&") + "&".join(f"{k}={v}" for k, v in params.items())
        sign = self._sign(ts, method.upper(), req_path, body_str)
        headers = {
            "CB-ACCESS-KEY": self.api_key,
            "CB-ACCESS-SIGN": sign,
            "CB-ACCESS-TIMESTAMP": ts,
            "CB-ACCESS-PASSPHRASE": self.passphrase,
            "Content-Type": "application/json",
        }
        resp = self._session.request(method.upper(), url, headers=headers, data=body_str if body_str else None, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data

    # Utilidades de precisión
    def _get_product(self, product_id: str) -> Dict[str, Any]:
        return self._request("GET", f"/products/{product_id.upper()}")

    @staticmethod
    def _round_step(value: float, step: float) -> float:
        if step <= 0:
            return float(value)
        # Evita issues binarios usando cuantización decimal-like simple
        n = int(round(value / step))
        return float(n * step)

    def connectivity_ok(self) -> bool:
        try:
            _ = self._request("GET", "/accounts")
            return True
        except Exception:  # noqa: BLE001
            return False

    def get_account(self) -> Dict[str, Any]:
        # Coinbase no expone equity igual que brokers; devolvemos stub útil
        accounts = self._request("GET", "/accounts")
        total_balance = 0.0
        for acc in accounts:
            try:
                total_balance += float(acc.get("balance", 0))
            except Exception:  # noqa: BLE001
                continue
        return {"equity": total_balance, "last_equity": total_balance, "unrealized_pl": 0.0}

    def place_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        type_: str = "market",
        price: Optional[float] = None,
        tif: str = "GTC",
        post_only: bool = False,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "product_id": symbol.upper(),
            "side": side.lower(),
            "type": type_.lower(),
        }
        # Ajustar precisión según producto
        try:
            prod = self._get_product(symbol)
            size_step = float(prod.get("base_increment") or 0)
            price_step = float(prod.get("quote_increment") or 0)
        except Exception:
            size_step = 0.0
            price_step = 0.0
        adj_qty = self._round_step(float(quantity), size_step) if size_step else float(quantity)
        adj_price = float(price) if price is not None else None
        if adj_price is not None and price_step:
            adj_price = self._round_step(adj_price, price_step)

        if type_.lower() == "market":
            payload["size"] = adj_qty
        elif type_.lower() == "limit":
            if price is None:
                raise ValueError("Se requiere price para órdenes limit en Coinbase")
            payload.update({"price": str(adj_price), "size": adj_qty, "time_in_force": tif.upper()})
            if post_only:
                payload["post_only"] = True
        else:
            payload["size"] = adj_qty
            if adj_price is not None:
                payload["price"] = str(adj_price)
        # Idempotencia: Coinbase acepta client_oid
        try:
            import uuid as _uuid
            payload.setdefault("client_oid", _uuid.uuid4().hex)
        except Exception:
            pass
        return self._request("POST", "/orders", json=payload)

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        _ = self._request("DELETE", f"/orders/{order_id}")
        return {"status": "canceled", "order_id": order_id}

    def cancel_all_orders(self, product_id: Optional[str] = None) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        if product_id:
            params["product_id"] = product_id.upper()
        data = self._request("DELETE", "/orders", params=params)
        return {"status": "ok", "canceled": len(data) if isinstance(data, list) else 0}

    def list_orders(self, status: str = "open", limit: int = 100) -> List[Dict[str, Any]]:
        params = {"status": status, "limit": limit}
        data = self._request("GET", "/orders", params=params)
        return list(data)

    def list_fills(self, product_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {"limit": limit}
        if product_id:
            params["product_id"] = product_id.upper()
        data = self._request("GET", "/fills", params=params)
        # Mapear fees a 'fee'
        for d in data:
            try:
                fee_val = d.get("fee") or d.get("liquidity_fee")
                if fee_val is not None:
                    d["fee"] = float(fee_val)
            except Exception:
                pass
        return list(data)

    # ===== Reauth/Backfill helpers (stubs útiles) =====
    def refresh_credentials(self, api_key: str | None = None, api_secret: str | None = None, passphrase: str | None = None) -> None:
        if api_key:
            self.api_key = api_key
        if api_secret:
            self.api_secret = api_secret
        if passphrase:
            self.passphrase = passphrase

    def backfill_recent(self, product_id: str, limit: int = 100) -> Dict[str, Any]:
        """Recupera órdenes y fills recientes para un producto y los devuelve (no persiste)."""
        orders = self._request("GET", "/orders", params={"status": "all", "product_id": product_id, "limit": limit})
        fills = self.list_fills(product_id=product_id, limit=limit)
        return {"orders": list(orders), "fills": list(fills)}


