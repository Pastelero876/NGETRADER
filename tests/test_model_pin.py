from __future__ import annotations

from nge_trader.services import model_session


def test_pin_unpin() -> None:
    model_session.pin("m1", "s1", "fh1")
    ms = model_session.get()
    assert ms.get("pinned") is True and ms.get("model_id") == "m1"
    model_session.unpin()
    ms2 = model_session.get()
    assert ms2.get("pinned") is False


