from __future__ import annotations

import json
import threading
from typing import Callable, Optional

import websocket  # type: ignore[import-not-found]


class AlpacaTradeWS:
    """Cliente simple de WebSocket para órdenes/actividades en Alpaca (stream de trading).

    Nota: Algunas cuentas requieren endpoints específicos y claves; este cliente es básico y
    orientado a actividades de trading. Ajusta la URL si usas data streaming.
    """

    def __init__(self, api_key: str, api_secret: str, on_message: Callable[[dict], None], on_event: Optional[Callable[[str], None]] = None) -> None:
        self.api_key = api_key
        self.api_secret = api_secret
        self.on_message = on_message
        self.on_event = on_event
        self.ws: websocket.WebSocketApp | None = None
        self.thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        url = "wss://paper-api.alpaca.markets/stream"

        def _on_open(ws):  # noqa: ANN001
            auth = {"action": "authenticate", "data": {"key_id": self.api_key, "secret_key": self.api_secret}}
            ws.send(json.dumps(auth))
            if self.on_event:
                self.on_event("ws_open")

        def _on_message(ws, message):  # noqa: ANN001
            try:
                data = json.loads(message)
                self.on_message(data)
            except Exception:
                pass

        def _on_error(ws, error):  # noqa: ANN001
            if self.on_event:
                self.on_event(f"ws_error: {error}")

        def _on_close(ws, close_status_code, close_msg):  # noqa: ANN001
            if self.on_event:
                self.on_event(f"ws_close: {close_status_code} {close_msg}")

        def _runner():
            backoff = 1.0
            while not self._stop.is_set():
                try:
                    self.ws = websocket.WebSocketApp(
                        url, on_open=_on_open, on_message=_on_message, on_error=_on_error, on_close=_on_close
                    )
                    self.ws.run_forever(ping_interval=20, ping_timeout=10)
                except Exception as exc:  # noqa: BLE001
                    if self.on_event:
                        self.on_event(f"ws_exception: {exc}")
                if self._stop.is_set():
                    break
                # reconexión con backoff exponencial
                import time

                time.sleep(backoff)
                backoff = min(backoff * 2.0, 30.0)
                if self.on_event:
                    self.on_event("ws_reconnect")

        self.thread = threading.Thread(target=_runner, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        try:
            self._stop.set()
            if self.ws:
                self.ws.close()
        except Exception:
            pass


