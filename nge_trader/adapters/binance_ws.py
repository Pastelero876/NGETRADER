from __future__ import annotations

import json
import threading
import time
import random
from typing import Any, Callable, Optional

import requests
from websocket import WebSocketApp  # type: ignore


class BinanceUserDataStream:
    """Gestor de User Data Stream con reconexión y keepalive.

    Permite registrar un callback para eventos de órdenes/ejecuciones.
    """

    def __init__(self, api_key: str, api_secret: str, base_url: Optional[str] = None, testnet: bool = False) -> None:
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url or ("https://testnet.binance.vision" if testnet else "https://api.binance.com")
        self.ws_base = "wss://stream.binance.com:9443" if not testnet else "wss://testnet.binance.vision"
        self._session = requests.Session()
        self._session.headers.update({"X-MBX-APIKEY": self.api_key})
        self._listen_key: Optional[str] = None
        self._ws: Optional[WebSocketApp] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._callback: Optional[Callable[[dict], None]] = None
        self._on_reconnect: Optional[Callable[[], None]] = None
        self._last_event_update: float | None = None

    # REST helpers para listenKey
    def _get_listen_key(self) -> str:
        url = f"{self.base_url}/api/v3/userDataStream"
        resp = self._session.post(url, timeout=10)
        resp.raise_for_status()
        return resp.json()["listenKey"]

    def _keepalive(self) -> None:
        if not self._listen_key:
            return
        try:
            url = f"{self.base_url}/api/v3/userDataStream"
            self._session.put(url, params={"listenKey": self._listen_key}, timeout=10)
        except Exception:
            pass

    def _build_ws_url(self) -> str:
        if not self._listen_key:
            raise RuntimeError("listenKey no inicializado")
        return f"{self.ws_base}/ws/{self._listen_key}"

    def _on_message(self, _ws: WebSocketApp, message: str) -> None:  # noqa: ANN001
        try:
            data = json.loads(message)
        except Exception:
            return
        try:
            # Guardar última marca de tiempo para backfill fino
            et = None
            if isinstance(data, dict):
                et = data.get("E") or data.get("eventTime") or data.get("T")
            if et:
                self._last_event_update = float(et) / (1000.0 if float(et) > 1e12 else 1.0)
        except Exception:
            pass
        if self._callback:
            try:
                self._callback(data)
            except Exception:
                pass

    def _on_error(self, _ws: WebSocketApp, _error: Any) -> None:  # noqa: ANN001, ANN401
        # Forzar cierre para que el bucle reconecte
        try:
            _ws.close()
        except Exception:
            pass

    def _on_close(self, _ws: WebSocketApp, _code: int, _msg: str) -> None:  # noqa: ANN001, ANN201
        # noop: el bucle principal maneja reconexiones
        return

    def _run_forever(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            try:
                if not self._listen_key:
                    self._listen_key = self._get_listen_key()
                    try:
                        # Notificar reconexión (rehydrate)
                        if self._on_reconnect:
                            self._on_reconnect()
                    except Exception:
                        pass
                # Tras reconectar, intentar backfill fino vía REST si existe marca de tiempo
                try:
                    if self._last_event_update:
                        # placeholder: aquí se podría invocar un callback externo o REST myTrades desde el adaptador de trading
                        pass
                except Exception:
                    pass
                url = self._build_ws_url()
                self._ws = WebSocketApp(
                    url,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )
                # Lanzar keepalive en paralelo
                last_keepalive = time.time()

                def _inner_run() -> None:
                    self._ws.run_forever(ping_interval=20, ping_timeout=10)  # type: ignore[attr-defined]

                t = threading.Thread(target=_inner_run, daemon=True)
                t.start()
                # Bucle de servicio
                while t.is_alive() and not self._stop.is_set():
                    now = time.time()
                    if now - last_keepalive > 30 * 60 - 30:
                        self._keepalive()
                        last_keepalive = now
                    time.sleep(0.5)
                # Si salimos, limpiar y reconectar
                try:
                    if self._ws:
                        self._ws.close()
                except Exception:
                    pass
                # Reset listenKey para renovar en próxima vuelta
                self._listen_key = None
                # Backoff progresivo
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


