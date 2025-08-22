from __future__ import annotations

import shutil
from pathlib import Path
from datetime import datetime

from nge_trader.repository.db import DB_PATH


def main() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    src = Path(DB_PATH)
    backups = Path("backups")
    backups.mkdir(parents=True, exist_ok=True)
    dst = backups / f"app_{ts}.db"
    if src.exists():
        shutil.copy2(src, dst)
    # Zip opcional
    zip_path = backups / f"app_{ts}.zip"
    shutil.make_archive(str(zip_path).removesuffix('.zip'), 'zip', backups, dst.name)
    return str(zip_path)


if __name__ == "__main__":
    out = main()
    print(out)

from __future__ import annotations

from pathlib import Path
import shutil


def main() -> None:
    src = Path("data/app.db")
    dst_dir = Path("data/backups")
    dst_dir.mkdir(parents=True, exist_ok=True)
    if src.exists():
        dst = dst_dir / f"app.db.bak"
        shutil.copy2(src, dst)
        print({"status": "ok", "backup": str(dst)})
    else:
        print({"status": "noop", "reason": "db not found"})


if __name__ == "__main__":
    main()


