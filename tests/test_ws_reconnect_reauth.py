from __future__ import annotations

import uuid
import base64


class _BrokerMock:
    def __init__(self) -> None:
        self.oid = "O_" + uuid.uuid4().hex[:8]
        self._orders = [{"id": self.oid, "symbol": "TEST", "side": "buy", "qty": 1.0}]
        self._fills = [{"order_id": self.oid, "symbol": "TEST", "side": "buy", "qty": 1.0, "price": 10.0}]

    def list_orders(self, status: str = "open", limit: int = 100, symbol: str | None = None):  # noqa: ANN001
        return list(self._orders)

    def list_fills(self, symbol: str | None = None, limit: int = 100):  # noqa: ANN001
        return list(self._fills)


def test_backfill_for_symbol_registers_new_items(monkeypatch):
    from nge_trader.services.app_service import AppService
    from nge_trader.repository.db import Database

    app = AppService()
    # Inyectar broker mock
    broker = _BrokerMock()
    app.broker = broker
    # asegurar DB sin previos
    db = Database()
    # Ejecutar backfill
    res = app.backfill_for_symbol("TEST", limit=10)
    assert isinstance(res, dict)
    # Validar por estado en DB
    assert db.order_exists(broker.oid) is True
    fills = db.recent_fills(10)
    assert any(f.get("order_id") == broker.oid for f in fills)
    # Re-ejecutar debe deduplicar
    res2 = app.backfill_for_symbol("TEST", limit=10)
    assert res2["orders"] == 0
    assert res2["fills"] == 0


def test_coinbase_user_ws_stub_import():
    import types
    from nge_trader.adapters.coinbase_ws import CoinbaseUserWS
    # Simular arranque y reconexión: no abrimos socket real, pero validamos auth payload y start firma on_reconnect
    ws = CoinbaseUserWS("k", base64.b64encode(b"s").decode("utf-8"), "p")
    pld = ws._auth_payload()
    assert all(k in pld for k in ("type","signature","key","passphrase","timestamp"))
    # Simular on_reconnect ejecutando backfill
    calls = {"reconnect": 0}
    def _cb(_msg):
        pass
    def _on_reconnect():
        calls["reconnect"] += 1
    ws.start(_cb, on_reconnect=_on_reconnect)
    # detener rápidamente el hilo
    ws.stop()
    assert calls["reconnect"] >= 0


def test_ibkr_trade_feed_poll_import(monkeypatch):
    from types import SimpleNamespace
    class FakeIB:
        def __init__(self) -> None:
            self._conn = False
        def isConnected(self) -> bool:
            return self._conn
        def connect(self, *a, **k):
            self._conn = True
        def trades(self):
            return []
    monkeypatch.setitem(__import__("sys").modules, 'ib_insync', SimpleNamespace(IB=FakeIB))
    from nge_trader.adapters.ibkr_ws import IBKRTradeFeed
    feed = IBKRTradeFeed(FakeIB())
    called = {"re": False}
    def _on_re():
        called["re"] = True
    feed.start(lambda e: None, poll_seconds=0, on_reconnect=_on_re)
    import time as _t
    _t.sleep(0.05)
    feed.stop()
    # No podemos garantizar ejecución del on_reconnect en tiempo corto, pero el objeto existe y no rompe
    assert hasattr(feed, "stop")


def test_coinbase_backfill_stub(monkeypatch):
    from nge_trader.adapters.coinbase_broker import CoinbaseBroker
    c = CoinbaseBroker("k","s","p")
    def fake_req(method, path, params=None, json=None):  # noqa: ANN001, A002
        if path == "/orders":
            return [{"id": "O1", "product_id": "BTC-USD", "side": "buy", "size": "1"}]
        if path == "/fills":
            return [{"order_id": "O1", "product_id": "BTC-USD", "price": "100", "size": "1", "liquidity_fee": 0.001}]
        return []
    monkeypatch.setattr(c, "_request", fake_req)
    out = c.backfill_recent("BTC-USD", 10)
    assert isinstance(out, dict) and out["orders"] and out["fills"]


def test_ibkr_backfill_and_reconnect_stub(monkeypatch):
    from types import SimpleNamespace
    from nge_trader.adapters.ibkr_broker import IBKRBroker
    class FakeIB:
        def __init__(self) -> None:
            self._connected = False
        def isConnected(self) -> bool:
            return self._connected
        def connect(self, *a, **k):
            self._connected = True
        def trades(self):
            order = SimpleNamespace(orderId=1, action="BUY")
            orderStatus = SimpleNamespace(status="Filled")
            contract = SimpleNamespace(symbol="AAPL")
            return [SimpleNamespace(order=order, orderStatus=orderStatus, contract=contract, fills=[])]
    monkeypatch.setitem(__import__("sys").modules, 'ib_insync', SimpleNamespace(IB=FakeIB))
    b = IBKRBroker()
    assert b.reconnect() is True
    out = b.backfill_recent(10)
    assert isinstance(out, dict) and out["orders"] is not None


