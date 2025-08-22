from __future__ import annotations

from nge_trader.repository.db import Database


def test_audit_log_chain_append_and_verify():
    db = Database()
    db.append_audit_log('{"event":"test","details":"ok"}')
    assert db.verify_audit_log(limit=10) is True


def test_audit_log_chain_integrity():
    db = Database()
    db.append_audit_log("{\"event\": \"start\"}")
    db.append_audit_log("{\"event\": \"next\"}")
    assert db.verify_audit_log(limit=10) is True


