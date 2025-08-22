def test_binance_fees_field_mapping(monkeypatch):
    from nge_trader.adapters.binance_trade import BinanceBroker
    b = BinanceBroker("k","s", testnet=True)
    def fake_get(path, params=None, signed=False):  # noqa: ANN001
        return [{"commission": "0.001", "price": "100", "qty": "1", "symbol": "BTCUSDT"}]
    monkeypatch.setattr(b, "_get", fake_get)
    data = b.list_fills("BTCUSDT", 1)
    assert data and float(data[0].get("fee") or 0.0) == 0.001

def test_coinbase_fees_field_mapping(monkeypatch):
    from nge_trader.adapters.coinbase_broker import CoinbaseBroker
    c = CoinbaseBroker("k","s","p")
    def fake_req(method, path, params=None, json=None):  # noqa: ANN001, A002
        return [{"liquidity_fee": 0.002, "price": "100", "size": "1", "product_id": "BTC-USD"}]
    monkeypatch.setattr(c, "_request", fake_req)
    data = c.list_fills("BTC-USD", 1)
    assert data and float(data[0].get("fee") or 0.0) == 0.002


def test_ibkr_fees_field_mapping(monkeypatch):
    from types import SimpleNamespace
    from nge_trader.adapters.ibkr_broker import IBKRBroker
    # Simular entorno ib_insync mínimo
    class FakeIB:
        def __init__(self) -> None:
            pass
        def isConnected(self) -> bool:
            return True
        def connect(self, *a, **k):
            return None
        def trades(self):
            exec_obj = SimpleNamespace(shares=1.0, price=100.0, time="2020-01-01T00:00:00Z")
            comrep = SimpleNamespace(commission=0.25)
            order = SimpleNamespace(orderId=123, action="SELL")
            orderStatus = SimpleNamespace(status="Filled", filled=1.0)
            contract = SimpleNamespace(symbol="AAPL")
            fill = SimpleNamespace(execution=exec_obj, commissionReport=comrep)
            return [SimpleNamespace(order=order, orderStatus=orderStatus, contract=contract, fills=[fill])]
    # Parchear import y objeto IB interno
    monkeypatch.setitem(__import__("sys").modules, 'ib_insync', SimpleNamespace(IB=FakeIB))
    b = IBKRBroker()
    data = b.list_fills()
    assert data and abs(float(data[0].get("fee") or 0.0) - 0.25) < 1e-9


def test_fees_schedule_api_roundtrip():
    from nge_trader.entrypoints.api import api_set_fee_schedule, api_get_fee_schedule, FeePayload
    p = FeePayload(exchange="BINANCE", symbol="ETHUSDT", maker_bps=1.1, taker_bps=6.2)
    res = api_set_fee_schedule(p)
    assert res.get("status") == "ok"
    g = api_get_fee_schedule("BINANCE", "ETHUSDT")
    assert g.get("row") and abs(float(g["row"]["taker_bps"]) - 6.2) < 1e-9


