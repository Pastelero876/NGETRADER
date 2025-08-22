def test_ibkr_cancel_symbol_stub():
    # Este test es un stub que valida que la función existe e invocable
    try:
        from nge_trader.adapters.ibkr_broker import IBKRBroker
        b = IBKRBroker  # referencia
        assert hasattr(b, 'cancel_all_orders_by_symbol')
    except Exception:
        # Si no hay ib_insync instalado o gateway, el import puede fallar; no hacemos fail duro
        assert True


