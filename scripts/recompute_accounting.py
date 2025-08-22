from __future__ import annotations

import os
from nge_trader.services.accounting import recompute_lot_accounting


def main() -> None:
    sym = os.environ.get("ACCOUNT_SYMBOL")
    stats = recompute_lot_accounting(symbol=sym)
    print(stats)


if __name__ == "__main__":
    main()


