from __future__ import annotations

from jsonschema import validate
PROFILE_WRAPPER_SCHEMA = {
    "type": "object",
    "properties": {
        "schema_version": {"type": "integer", "minimum": 1},
        "profiles": {"type": "object"},
    },
    "required": ["schema_version", "profiles"],
}



SLICING_PROFILE_SCHEMA = {
    "type": "object",
    "additionalProperties": {
        "type": "object",
        "properties": {
            "slicing_tranches": {"type": "integer", "minimum": 1},
            "slicing_bps": {"type": "number", "minimum": 0},
        },
        "additionalProperties": False,
    },
}

RISK_PROFILE_SCHEMA = {
    "type": "object",
    "additionalProperties": {
        "type": "object",
        "properties": {
            "risk_pct_per_trade": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "additionalProperties": False,
    },
}


def validate_json_profile(payload: str, schema: dict) -> None:
    import json
    data = json.loads(payload)
    validate(instance=data, schema=schema)


def migrate_profile(payload: str) -> str:
    """Migra perfiles JSON a última versión del esquema si es necesario.

    Estrategia: si no tiene `schema_version`, envolvemos en {schema_version:1, profiles:<data>}.
    """
    import json
    data = json.loads(payload)
    if not isinstance(data, dict) or "schema_version" not in data:
        wrapped = {"schema_version": 1, "profiles": data}
        validate(instance=wrapped, schema=PROFILE_WRAPPER_SCHEMA)
        return json.dumps(wrapped)
    # Futuras migraciones por versión
    return payload


