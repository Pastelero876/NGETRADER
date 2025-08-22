from nge_trader.services.oms import generate_idempotency_key


def test_generate_idempotency_key_unique():
    k1 = generate_idempotency_key("BTCUSDT", "buy")
    k2 = generate_idempotency_key("BTCUSDT", "buy")
    assert k1 != k2
    assert k1.startswith("BTCUSDT_buy_")


