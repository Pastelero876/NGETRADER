from __future__ import annotations

import time
import sys
import requests


def main() -> int:
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000/health"
    interval = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    fail_count = 0
    while True:
        try:
            r = requests.get(url, timeout=3)
            if r.ok:
                fail_count = 0
            else:
                fail_count += 1
        except Exception:
            fail_count += 1
        if fail_count >= 3:
            print({"status": "restart_recommended", "url": url, "ts": time.time()})
            fail_count = 0
        time.sleep(interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


