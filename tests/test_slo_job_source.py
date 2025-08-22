from __future__ import annotations


def test_p95_py_calc() -> None:
    from scripts.update_slo_from_tca import p95
    assert p95([1, 2, 3, 4, 5]) in (4.0, 5.0)


def test_promql_build() -> None:
    from scripts.update_slo_from_tca import compute_p95_prom

    s, lat = compute_p95_prom("BTCUSDT", 7)
    assert (s is None) or isinstance(s, float)
    assert (lat is None) or isinstance(lat, float)


