from __future__ import annotations

import argparse
import json
from pathlib import Path

from nge_trader.services.templates import migrate_profiles_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrar perfiles JSON al esquema actual (envuelto con schema_version)")
    parser.add_argument("input", type=Path, help="Ruta del archivo JSON de entrada")
    parser.add_argument("--output", "-o", type=Path, default=None, help="Ruta del archivo de salida (por defecto, <input>.migrated.json)")
    args = parser.parse_args()

    raw = args.input.read_text(encoding="utf-8")
    migrated = migrate_profiles_payload(raw)

    out_path = args.output or args.input.with_suffix(".migrated.json")
    out_path.write_text(json.dumps(json.loads(migrated), ensure_ascii=False, indent=2), encoding="utf-8")
    print(str(out_path))


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import sys


def migrate(payload: str) -> str:
    data = json.loads(payload)
    # Stub: asegura schema_version y campos requeridos
    data.setdefault("schema_version", 1)
    return json.dumps(data, ensure_ascii=False)


def main() -> None:
    raw = sys.stdin.read()
    print(migrate(raw))


if __name__ == "__main__":
    main()


