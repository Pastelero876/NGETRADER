def test_ui_export_button_imports():
    from nge_trader.entrypoints.desktop import DesktopApp
    assert hasattr(DesktopApp, '_export_daily_report')


def test_api_status_structure():
    from nge_trader.entrypoints.api import api_status
    data = api_status()
    assert isinstance(data, dict)
    assert 'kill_switch_armed' in data and 'paused_symbols' in data and 'role' in data


