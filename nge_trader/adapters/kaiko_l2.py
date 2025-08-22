from __future__ import annotations

import json
import threading
import time
from typing import Callable, Optional

import websocket


class KaikoL2WS:
    """Cliente WS mínimo para Kaiko L2 (esqueleto/compatible con tests offline).

    - No abre conexión real si no hay API key; permite inyectar eventos para pruebas.
    - on_message recibe dict con: { 'type': 'depth', 'symbol', 'bids': [[p,q],...], 'asks': [[p,q],...], 'event_time': epoch_ms }
    """

    def __init__(self, api_key: Optional[str]) -> None:
        self.api_key = api_key or None
        self._ws: websocket.WebSocketApp | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self, symbol: str, on_message: Callable[[dict], None]) -> None:
        # En modo stub si no hay API key
        if not self.api_key:
            def _stub() -> None:
                while not self._stop.is_set():
                    time.sleep(1)
            self._thread = threading.Thread(target=_stub, daemon=True)
            self._thread.start()
            return
        url = "wss://us.market-data-api.kaiko.io/v2/update"
        headers = [f"Authorization: Bearer {self.api_key}"]
        def _on_message(_ws: websocket.WebSocketApp, message: str) -> None:  # noqa: ANN001
            try:
                evt = json.loads(message)
                if isinstance(evt, dict) and evt.get("type") == "depth":
                    on_message(evt)
            except Exception:
                pass
        def _on_open(ws: websocket.WebSocketApp) -> None:  # noqa: ANN001
            try:
                sub = {"type": "subscribe", "channel": "l2_book", "symbol": symbol}
                ws.send(json.dumps(sub))
            except Exception:
                pass
        self._ws = websocket.WebSocketApp(url, on_message=_on_message, header=headers, on_open=_on_open)
        def _run() -> None:
            while not self._stop.is_set():
                try:
                    self._ws.run_forever(ping_interval=20, ping_timeout=10)
                except Exception:
                    time.sleep(2)
        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        try:
            if self._ws:
                self._ws.close()
        except Exception:
            pass

