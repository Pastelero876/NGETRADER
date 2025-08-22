from __future__ import annotations

import threading
import time
from typing import Callable, Optional
import queue


class IBKRTradeFeed:
    """Poller simple que emite actualizaciones de trades/órdenes de IBKR vía callback.

    ib_insync entrega eventos en objetos; aquí simulamos un feed periódico que recorre trades()
    y emite un dict por cada trade nuevo visto.
    """

    def __init__(self, ib) -> None:  # noqa: ANN001
        self.ib = ib
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._callback: Optional[Callable[[dict], None]] = None
        self._seen: set[int] = set()
        self._sim_queue: "queue.Queue[dict]" = queue.Queue()

    def start(self, callback: Callable[[dict], None], poll_seconds: int = 2, on_reconnect: Optional[Callable[[], None]] = None) -> None:
        self._callback = callback
        self._stop.clear()

        def _worker() -> None:
            # was_connected flag no usado; eliminar para evitar warning
            while not self._stop.is_set():
                try:
                    # reconexión si se pierde conexión
                    try:
                        if not self.ib.isConnected():
                            self.ib.connect("127.0.0.1", 7497, clientId=1, readonly=False, timeout=10)
                            if on_reconnect:
                                try:
                                    on_reconnect()
                                except Exception:
                                    pass
                    except Exception:
                        pass
                    # Simular eventos si se envían por cola
                    try:
                        while True:
                            evt = self._sim_queue.get_nowait()
                            if self._callback:
                                self._callback(evt)
                    except queue.Empty:
                        pass
                    for t in self.ib.trades():
                        oid = getattr(t.order, "orderId", None)
                        if oid is None or oid in self._seen:
                            continue
                        self._seen.add(int(oid))
                        evt = {
                            "type": "ibkr_trade",
                            "order_id": oid,
                            "status": getattr(t.orderStatus, "status", None),
                            "symbol": getattr(getattr(t, "contract", None), "symbol", None),
                        }
                        try:
                            if self._callback:
                                self._callback(evt)
                        except Exception:
                            pass
                except Exception:
                    pass
                time.sleep(max(int(poll_seconds), 1))

        self._thread = threading.Thread(target=_worker, daemon=True)
        self._thread.start()

    # Simulación de eventos (fills/cancels)
    def push_simulated_event(self, evt: dict) -> None:
        self._sim_queue.put(dict(evt))

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)


