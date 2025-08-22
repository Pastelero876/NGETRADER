from nge_trader.services.app_service import AppService


def test_reconcile_state_runs():
    svc = AppService()
    diffs = svc.reconcile_state()
    assert isinstance(diffs, dict)
    assert "missing_in_db" in diffs and "missing_in_broker" in diffs


def test_reconcile_with_resolve():
    svc = AppService()
    diffs = svc.reconcile_state(resolve=True)
    assert isinstance(diffs, dict)


