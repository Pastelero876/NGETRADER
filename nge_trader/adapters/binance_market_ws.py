from __future__ import annotations

import json
import threading
import time
from typing import Callable, Optional

from websocket import WebSocketApp  # type: ignore
from nge_trader.services.net_sim import maybe_jitter_and_drop
from nge_trader.services.metrics import set_metric_labeled


class BinanceMarketWS:
    """WS de mercado para Binance: aggTrade o miniTicker por símbolo.

    Callback recibe dict con al menos: { 'type': 'trade'|'mini', 'symbol', 'price', 'event_time' }
    """

    def __init__(self, symbol: str, testnet: bool = False, stream: str = "aggTrade", depth_levels: int = 5) -> None:
        self.symbol = symbol.lower()
        self.stream = stream
        self.ws_base = "wss://stream.binance.com:9443" if not testnet else "wss://testnet.binance.vision"
        self._ws: Optional[WebSocketApp] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._callback: Optional[Callable[[dict], None]] = None
        self._depth_levels = int(depth_levels)

    def _build_url(self) -> str:
        # Soporta aggTrade, miniTicker, bookTicker y depth configurable
        if self.stream == "aggTrade":
            stream = "@aggTrade"
        elif self.stream == "miniTicker":
            stream = "@miniTicker"
        elif self.stream.startswith("depth"):
            try:
                lvl = int(self.stream.replace("depth", "") or self._depth_levels)
            except Exception:
                lvl = self._depth_levels
            stream = f"@depth{max(5, min(1000, int(lvl)))}@100ms"
        elif self.stream == "depth":
            stream = f"@depth{max(5, min(1000, int(self._depth_levels)))}@100ms"
        else:
            stream = "@bookTicker"
        return f"{self.ws_base}/ws/{self.symbol}{stream}"

    def _on_message(self, _ws: WebSocketApp, message: str) -> None:  # noqa: ANN001
        try:
            # Simulación de red (opcional)
            if maybe_jitter_and_drop():
                return
            data = json.loads(message)
        except Exception:
            return
        evt: dict
        if "e" in data and data.get("e") == "aggTrade":
            evt = {
                "type": "trade",
                "symbol": data.get("s"),
                "price": float(data.get("p") or 0.0),
                "qty": float(data.get("q") or 0.0),
                "event_time": int(data.get("E") or 0),
            }
        elif "e" in data and data.get("e") == "24hrMiniTicker":
            evt = {
                "type": "mini",
                "symbol": data.get("s"),
                "price": float(data.get("c") or 0.0),
                "event_time": int(data.get("E") or 0),
            }
        elif ("s" in data) and ("b" in data) and ("a" in data):
            # bookTicker o depth según stream configurado
            if self.stream.startswith("depth") or self.stream == "depth":
                def _to_pairs(arr):
                    try:
                        return [(float(p), float(q)) for p, q in arr]
                    except Exception:
                        return []
                evt = {
                    "type": "depth",
                    "symbol": data.get("s"),
                    "bids": _to_pairs(data.get("b") or data.get("bids") or []),
                    "asks": _to_pairs(data.get("a") or data.get("asks") or []),
                    "event_time": int(data.get("E") or 0),
                }
            else:
                evt = {
                    "type": "book",
                    "symbol": data.get("s"),
                    "bid": float(data.get("b") or 0.0),
                    "ask": float(data.get("a") or 0.0),
                    "bid_size": float(data.get("B") or 0.0) if data.get("B") is not None else None,
                    "ask_size": float(data.get("A") or 0.0) if data.get("A") is not None else None,
                    "event_time": int(data.get("E") or 0),
                }
        else:
            return
        if self._callback:
            try:
                self._callback(evt)
            except Exception:
                pass
        # Empujar a LowLatencyProvider si es trade/book/depth
        try:
            from nge_trader.adapters.lowlat_provider import LowLatencyProvider
            ll = LowLatencyProvider()
            if evt.get("type") == "trade":
                ll.push_event(evt.get("symbol"), float(evt.get("price") or 0.0), float(evt.get("qty") or 0.0), ts=float(evt.get("event_time") or 0.0))
                # VPIN actualizado con nuevos trades
                try:
                    sym = str(evt.get("symbol") or "").upper()
                    vpin = ll.get_vpin(sym)
                    if vpin is not None:
                        set_metric_labeled("vpin", float(vpin), {"symbol": sym})
                except Exception:
                    pass
            elif evt.get("type") == "book":
                ll.push_quote(
                    evt.get("symbol"),
                    float(evt.get("bid") or 0.0),
                    float(evt.get("ask") or 0.0),
                    float(evt.get("bid_size") or 0.0) if evt.get("bid_size") is not None else None,
                    float(evt.get("ask_size") or 0.0) if evt.get("ask_size") is not None else None,
                    ts=float(evt.get("event_time") or 0.0),
                )
                # Exponer métricas L1 (microprice/ofi)
                try:
                    sym = str(evt.get("symbol") or "").upper()
                    mp = ll.get_microprice(sym) or 0.0
                    ofi, _obi = ll.get_ofi_obi(sym)
                    set_metric_labeled("l1_microprice", float(mp), {"symbol": sym})
                    set_metric_labeled("l1_ofi", float(ofi), {"symbol": sym})
                except Exception:
                    pass
            elif evt.get("type") == "depth":
                ll.push_depth(
                    evt.get("symbol"),
                    list(evt.get("bids") or []),
                    list(evt.get("asks") or []),
                    ts=float(evt.get("event_time") or 0.0),
                )
                # Exponer métricas L2 (microprice/ofi por niveles)
                try:
                    sym = str(evt.get("symbol") or "").upper()
                    mp2 = ll.get_l2_microprice(sym) or 0.0
                    ofi2 = ll.get_l2_ofi(sym)
                    set_metric_labeled("l2_microprice", float(mp2), {"symbol": sym})
                    set_metric_labeled("l2_ofi", float(ofi2), {"symbol": sym})
                except Exception:
                    pass
        except Exception:
            pass

    def _on_error(self, _ws: WebSocketApp, _error) -> None:  # noqa: ANN001, ANN401
        try:
            _ws.close()
        except Exception:
            pass

    def _run_forever(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            try:
                url = self._build_url()
                self._ws = WebSocketApp(url, on_message=self._on_message, on_error=self._on_error)
                self._ws.run_forever(ping_interval=20, ping_timeout=10)  # type: ignore[attr-defined]
            except Exception:
                pass
            # reconexión con backoff
            time.sleep(backoff)
            backoff = min(backoff * 2, 30)

    def start(self, callback: Callable[[dict], None]) -> None:
        self._callback = callback
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


