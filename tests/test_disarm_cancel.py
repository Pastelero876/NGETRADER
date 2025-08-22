from __future__ import annotations

import os

from nge_trader.services.control import disarm_and_cancel_if_config


def test_disarm_cancel_flag() -> None:
    os.environ["KILL_CANCEL_PENDING"] = "true"
    out = disarm_and_cancel_if_config()
    assert isinstance(out, dict)
    assert out.get("did_cancel") is True
    assert isinstance(out.get("details"), list)


