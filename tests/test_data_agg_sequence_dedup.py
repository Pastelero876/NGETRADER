from __future__ import annotations

import pandas as pd

from nge_trader.services.data_agg import deduplicate_and_order_events, stable_agg_trades
from nge_trader.adapters.lowlat_provider import LowLatencyProvider


def test_deduplicate_and_order_by_sequence():
    df = pd.DataFrame([
        {"sequence_id": 2, "ts": 3, "price": 11, "qty": 1},
        {"sequence_id": 1, "ts": 1, "price": 10, "qty": 2},
        {"sequence_id": 2, "ts": 2, "price": 12, "qty": 1},  # duplicado sequence_id=2, debe quedarse el último por ts
    ])
    out = deduplicate_and_order_events(df)
    assert list(out["sequence_id"]) == [1, 2]
    assert float(out.iloc[1]["price"]) == 12.0


def test_stable_agg_trades_vwap():
    df = pd.DataFrame([
        {"price": 10.0, "qty": 1.0, "sequence_id": 1},
        {"price": 11.0, "qty": 2.0, "sequence_id": 2},
        {"price": 12.0, "qty": 3.0, "sequence_id": 3},
    ])
    out = stable_agg_trades(df)
    assert "vwap20" in out.columns and "vol20" in out.columns
    # VWAP total en último punto = (10*1 + 11*2 + 12*3) / (1+2+3) = 68/6 = 11.3333...
    assert abs(float(out.iloc[-1]["vwap20"]) - (68.0 / 6.0)) < 1e-9


def test_lowlat_provider_vwap_and_vol():
    ll = LowLatencyProvider(max_events=10)
    for i, (p, q) in enumerate([(10,1),(11,2),(12,3)], start=1):
        ll.push_event("BTCUSDT", price=p, qty=q, sequence_id=i, ts=i)
    vwap = ll.get_vwap("BTCUSDT")
    vol = ll.get_volatility("BTCUSDT", window=3)
    assert vwap is not None and vol is not None and vwap > 0

