from __future__ import annotations

from typing import Any, Dict, List, Literal
import time
import random

import requests


class AlpacaBroker:
    """Adaptador mínimo para Alpaca Markets (paper o live según credenciales)."""

    def __init__(self, api_key: str, api_secret: str, base_url: str | None = None) -> None:
        self.api_key = api_key
        self.api_secret = api_secret
        # Paper por defecto
        self.base_url = base_url or "https://paper-api.alpaca.markets"
        self._session = requests.Session()
        self._session.headers.update(
            {
                "APCA-API-KEY-ID": self.api_key,
                "APCA-API-SECRET-KEY": self.api_secret,
            }
        )

    def _request_with_retry(self, method: str, path: str, json_body: Dict[str, Any] | None = None) -> Any:
        url = self.base_url + path
        for attempt in range(5):
            try:
                if method == "GET":
                    resp = self._session.get(url, timeout=30)
                elif method == "POST":
                    resp = self._session.post(url, json=json_body, timeout=30)
                elif method == "DELETE":
                    resp = self._session.delete(url, timeout=30)
                elif method == "PATCH":
                    resp = self._session.patch(url, json=json_body, timeout=30)
                else:
                    raise ValueError("Método HTTP no soportado")
                resp.raise_for_status()
                return resp.json() if resp.content else {}
            except Exception:
                time.sleep(0.5 * (2 ** attempt) + random.random() * 0.2)
        # último intento
        if method == "GET":
            resp = self._session.get(url, timeout=30)
        elif method == "POST":
            resp = self._session.post(url, json=json_body, timeout=30)
        elif method == "DELETE":
            resp = self._session.delete(url, timeout=30)
        else:
            resp = self._session.patch(url, json=json_body, timeout=30)
        resp.raise_for_status()
        return resp.json() if resp.content else {}

    def _get(self, path: str) -> Any:
        return self._request_with_retry("GET", path)

    def _post(self, path: str, json: Dict[str, Any]) -> Any:
        return self._request_with_retry("POST", path, json_body=json)

    def get_account(self) -> Dict[str, Any]:
        return self._get("/v2/account")

    def list_positions(self) -> List[Dict[str, Any]]:
        data = self._get("/v2/positions")
        return list(data)

    def place_order(
        self,
        symbol: str,
        side: Literal["buy", "sell"],
        quantity: float,
        type_: str = "market",
        tif: str = "day",  # day, ioc, fok, gtc
        client_order_id: str | None = None,
    ) -> Dict[str, Any]:
        payload = {
            "symbol": symbol,
            "qty": quantity,
            "side": side,
            "type": type_,
            "time_in_force": tif,
        }
        if client_order_id:
            payload["client_order_id"] = client_order_id
        return self._post("/v2/orders", json=payload)

    def place_bracket_order(
        self,
        symbol: str,
        side: Literal["buy", "sell"],
        quantity: float,
        take_profit_price: float,
        stop_loss_price: float,
        type_: str = "market",
        tif: str = "gtc",
    ) -> Dict[str, Any]:
        payload = {
            "symbol": symbol,
            "qty": quantity,
            "side": side,
            "type": type_,
            "time_in_force": tif,
            "order_class": "bracket",
            "take_profit": {"limit_price": round(float(take_profit_price), 4)},
            "stop_loss": {"stop_price": round(float(stop_loss_price), 4)},
        }
        return self._post("/v2/orders", json=payload)

    def list_fills(self) -> List[Dict[str, Any]]:
        """Obtiene actividades de tipo FILL (ejecuciones)."""
        # Algunas cuentas usan data API, pero /v2/account/activities suele funcionar en trading API
        data = self._get("/v2/account/activities?activity_types=FILL")
        # Añadir 'fee' si hay 'commission' o similar
        out: List[Dict[str, Any]] = []
        for d in data:
            try:
                if "fee" not in d:
                    fee_val = d.get("commission") or d.get("fees") or 0.0
                    d["fee"] = float(fee_val or 0.0)
            except Exception:
                d["fee"] = 0.0
            out.append(d)
        return out

    def list_orders(self, status: str = "open", limit: int = 100) -> List[Dict[str, Any]]:
        data = self._get(f"/v2/orders?status={status}&limit={limit}")
        return list(data)

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        self._request_with_retry("DELETE", f"/v2/orders/{order_id}")
        return {"status": "canceled", "order_id": order_id}

    def replace_order(self, order_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request_with_retry("PATCH", f"/v2/orders/{order_id}", json_body=payload)

    def close_all_positions(self) -> Dict[str, Any]:
        return self._request_with_retry("DELETE", "/v2/positions")

    def close_symbol_position(self, symbol: str) -> Dict[str, Any]:
        return self._request_with_retry("DELETE", f"/v2/positions/{symbol}")

    def cancel_all_orders(self) -> Dict[str, Any]:
        return self._request_with_retry("DELETE", "/v2/orders")

    def connectivity_ok(self) -> bool:
        try:
            _ = self.get_account()
            return True
        except Exception:
            return False


