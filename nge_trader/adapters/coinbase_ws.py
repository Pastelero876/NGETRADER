from __future__ import annotations

import json
import threading
import time
import random
from typing import Any, Callable, Optional
import queue

from websocket import WebSocketApp  # type: ignore
import base64
import hashlib
import hmac


class CoinbaseUserWS:
    """User WebSocket para Coinbase Exchange (stub funcional con reconexión y callback).

    Intenta suscribirse a canales privados de usuario. La autenticación exacta puede variar
    según versión del feed; aquí enviamos un mensaje estándar y dejamos la simulación en tests.
    """

    def __init__(self, api_key: str, api_secret: str, passphrase: str, url: str | None = None) -> None:
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        self.url = url or "wss://ws-feed.exchange.coinbase.com"
        self._ws: Optional[WebSocketApp] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._callback: Optional[Callable[[dict], None]] = None
        self._on_reconnect: Optional[Callable[[], None]] = None
        self._sim_queue: "queue.Queue[dict]" = queue.Queue()
        self._sim_thread: Optional[threading.Thread] = None

    def _on_message(self, _ws: WebSocketApp, message: str) -> None:  # noqa: ANN001
        try:
            data = json.loads(message)
        except Exception:
            return
        if self._callback:
            try:
                self._callback(data)
            except Exception:
                pass

    def _on_error(self, _ws: WebSocketApp, _error: Any) -> None:  # noqa: ANN001, ANN401
        try:
            _ws.close()
        except Exception:
            pass

    def _auth_payload(self) -> dict:
        ts = str(int(time.time()))
        request_path = "/users/self/verify"
        prehash = f"{ts}GET{request_path}".encode("utf-8")
        secret = base64.b64decode(self.api_secret)
        signature = hmac.new(secret, prehash, hashlib.sha256).digest()
        sign_b64 = base64.b64encode(signature).decode("utf-8")
        return {
            "type": "subscribe",
            "channels": ["user"],
            "signature": sign_b64,
            "key": self.api_key,
            "passphrase": self.passphrase,
            "timestamp": ts,
        }

    # ====== Simulación de eventos (cancel/replace/partial fills) ======
    def push_simulated_event(self, evt: dict) -> None:
        self._sim_queue.put(dict(evt))

    def simulate_new_order(self, order_id: str, product_id: str, side: str, size: float, price: float | None = None) -> None:
        evt = {
            "type": "received",
            "order_id": order_id,
            "product_id": product_id,
            "side": side,
            "size": str(size),
            **({"price": str(price)} if price is not None else {}),
        }
        self.push_simulated_event(evt)

    def simulate_partial_fill(self, order_id: str, product_id: str, size: float, price: float, liquidity: str = "T") -> None:
        evt = {
            "type": "match",
            "order_id": order_id,
            "product_id": product_id,
            "size": str(size),
            "price": str(price),
            "liquidity": liquidity,
        }
        self.push_simulated_event(evt)

    def simulate_replace(self, order_id: str, new_price: float) -> None:
        evt = {
            "type": "change",
            "order_id": order_id,
            "new_price": str(new_price),
        }
        self.push_simulated_event(evt)

    def simulate_cancel(self, order_id: str, product_id: str) -> None:
        evt = {
            "type": "done",
            "order_id": order_id,
            "product_id": product_id,
            "reason": "canceled",
        }
        self.push_simulated_event(evt)

    def _run_forever(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            try:
                self._ws = WebSocketApp(
                    self.url,
                    on_message=self._on_message,
                    on_error=self._on_error,
                )
                def _inner() -> None:
                    self._ws.run_forever(ping_interval=20, ping_timeout=10)  # type: ignore[attr-defined]
                t = threading.Thread(target=_inner, daemon=True)
                t.start()
                # Intento de suscripción (user)
                try:
                    sub = self._auth_payload()
                    time.sleep(0.1)
                    if self._ws and self._ws.sock and self._ws.sock.connected:  # type: ignore[attr-defined]
                        self._ws.send(json.dumps(sub))
                        # on_reconnect hook
                        if self._on_reconnect:
                            try:
                                self._on_reconnect()
                            except Exception:
                                pass
                        # Lanzar simulador de eventos si hay cola
                        if self._sim_thread is None or not self._sim_thread.is_alive():
                            def _drain() -> None:
                                while not self._stop.is_set() and (self._ws and self._ws.sock and self._ws.sock.connected):  # type: ignore[attr-defined]
                                    try:
                                        evt = self._sim_queue.get(timeout=0.2)
                                        if self._callback and isinstance(evt, dict):
                                            try:
                                                self._callback(evt)
                                            except Exception:
                                                pass
                                    except queue.Empty:
                                        pass
                            self._sim_thread = threading.Thread(target=_drain, daemon=True)
                            self._sim_thread.start()
                except Exception:
                    pass
                while t.is_alive() and not self._stop.is_set():
                    time.sleep(0.5)
                try:
                    if self._ws:
                        self._ws.close()
                except Exception:
                    pass
                time.sleep(backoff + random.random() * 0.5)
                backoff = min(backoff * 2, 30)
            except Exception:
                time.sleep(backoff + random.random() * 0.5)
                backoff = min(backoff * 2, 30)

    def start(self, callback: Callable[[dict], None], on_reconnect: Optional[Callable[[], None]] = None) -> None:
        self._callback = callback
        self._on_reconnect = on_reconnect
        self._stop.clear()
        self._thread = threading.Thread(target=self._run_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        try:
            if self._ws:
                self._ws.close()
        except Exception:
            pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)


