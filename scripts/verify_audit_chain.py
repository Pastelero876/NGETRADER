from __future__ import annotations

from nge_trader.repository.db import Database


def main(limit: int = 5000) -> bool:
    return Database().verify_audit_log(limit=limit)


if __name__ == "__main__":
    ok = main()
    print({"valid": ok})


