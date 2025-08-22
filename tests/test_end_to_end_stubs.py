def test_end_to_end_stubs():
    # Este test asegura que scripts clave existen e importan sin error
    import importlib
    for mod in [
        'scripts.export_daily_report',
        'scripts.recompute_accounting',
    ]:
        importlib.import_module(mod)
    assert True


def test_api_status_and_summary_imports():
    from nge_trader.entrypoints.api import api_status, api_summary
    st = api_status()
    sm = api_summary()
    assert isinstance(st, dict) and isinstance(sm, dict)
    assert "metrics" in sm and "slo_error_rate_recent" in sm["metrics"]
    assert "rate_limit_remaining" in sm["metrics"] and "rate_limit_capacity" in sm["metrics"]


def test_2fa_endpoints_smoke():
    from nge_trader.entrypoints.api import api_2fa_setup, api_2fa_verify
    from types import SimpleNamespace
    res = api_2fa_setup(None, account="test", issuer="NGE")
    assert res.get("status") == "ok" and res.get("secret") and res.get("otpauth_uri")
    secret_b32 = res["secret"]
    from nge_trader.services.secret_store import SecretStore
    code = SecretStore().totp_code(secret_b32)
    out = api_2fa_verify(SimpleNamespace(code=code))
    assert out.get("status") == "ok"


