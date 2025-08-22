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


