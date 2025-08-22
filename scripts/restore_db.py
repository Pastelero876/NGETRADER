from __future__ import annotations

from pathlib import Path
import shutil


def main() -> None:
    src = Path("data/backups/app.db.bak")
    dst = Path("data/app.db")
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print({"status": "ok", "restored": str(dst)})
    else:
        print({"status": "noop", "reason": "backup not found"})


if __name__ == "__main__":
    main()


