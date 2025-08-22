from nge_trader.services.app_service import AppService


def test_positions_snapshot():
    svc = AppService()
    _ = svc.get_portfolio_positions()
    # si no hay broker con posiciones, al menos no debe fallar
    assert True


