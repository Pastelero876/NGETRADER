from nge_trader.services.app_service import AppService
from nge_trader.repository.db import Database


def test_balance_snapshot_runs():
    svc = AppService()
    _ = svc.get_account_summary()
    rows = Database().recent_balances(5)
    assert isinstance(rows, list)


