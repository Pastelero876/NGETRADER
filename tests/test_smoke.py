def test_smoke_imports():
    import nge_trader  # noqa: F401
    from nge_trader.entrypoints.desktop import DesktopApp  # noqa: F401
    from nge_trader.services.backtester import ExecutedSignalBacktester  # noqa: F401
    assert True


