import importlib


def main() -> None:
    modules = [
        "nge_trader.adapters.binance_trade",
        "nge_trader.adapters.coinbase_broker",
        "nge_trader.adapters.resilient",
        "nge_trader.adapters.ibkr_broker",
        "nge_trader.config.settings",
        "nge_trader.services.oms",
        "nge_trader.services.app_service",
    ]
    for m in modules:
        importlib.import_module(m)
        print(f"IMPORTED: {m}")
    print("OK")


if __name__ == "__main__":
    main()


