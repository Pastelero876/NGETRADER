from __future__ import annotations

import json

from nge_trader.services.schema import validate_json_profile, SLICING_PROFILE_SCHEMA, RISK_PROFILE_SCHEMA, migrate_profile, PROFILE_WRAPPER_SCHEMA


def test_validate_slicing_profile_ok():
    payload = json.dumps({"BTCUSDT": {"slicing_tranches": 4, "slicing_bps": 5}})
    validate_json_profile(payload, SLICING_PROFILE_SCHEMA)


def test_validate_risk_profile_ok():
    payload = json.dumps({"BTCUSDT": {"risk_pct_per_trade": 0.05}})
    validate_json_profile(payload, RISK_PROFILE_SCHEMA)


def test_migrate_profile_wraps_schema_version():
    legacy = json.dumps({"slicing": {"BTCUSDT": {"slicing_tranches": 4, "slicing_bps": 5}}})
    wrapped = migrate_profile(legacy)
    obj = json.loads(wrapped)
    # Validate wrapper schema
    validate_json_profile(json.dumps(obj), PROFILE_WRAPPER_SCHEMA)
    assert obj.get("schema_version") == 1
    assert "profiles" in obj


