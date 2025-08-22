from nge_trader.services.ticker_ner import extract_tickers


def test_extract_tickers_basic():
    text = "$AAPL beats earnings; MSFT up as well."
    ticks = extract_tickers(text)
    assert "AAPL" in ticks
    assert "MSFT" in ticks


