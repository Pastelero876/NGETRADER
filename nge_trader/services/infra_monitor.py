from __future__ import annotations

import threading
import time
import psutil
import requests

from nge_trader.services.metrics import set_metric


class InfraMonitor:
    def __init__(self, probe_url: str | None = None, interval_sec: int = 15) -> None:
        self.probe_url = probe_url
        self.interval_sec = int(interval_sec)
        self._stop = threading.Event()
        self._t: threading.Thread | None = None

    def start(self) -> None:
        if self._t and self._t.is_alive():
            return
        self._stop.clear()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def stop(self) -> None:
        self._stop.set()
        if self._t and self._t.is_alive():
            self._t.join(timeout=3)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                cpu = float(psutil.cpu_percent(interval=None))
                mem = float(psutil.virtual_memory().percent)
                set_metric("infra_cpu_pct", cpu)
                set_metric("infra_mem_pct", mem)
            except Exception:
                pass
            if self.probe_url:
                try:
                    t0 = time.perf_counter()
                    requests.head(self.probe_url, timeout=3)
                    dt = (time.perf_counter() - t0) * 1000.0
                    set_metric("infra_rtt_ms", float(dt))
                except Exception:
                    pass
            time.sleep(self.interval_sec)


