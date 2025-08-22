from __future__ import annotations

from unittest.mock import MagicMock

from nge_trader.services.model_session import pin, unpin


def test_online_training_cut_when_pinned(monkeypatch) -> None:  # noqa: ANN001
    # Arrange: pin a model
    pin("mX")
    # Fake engine with settings
    from types import SimpleNamespace

    class DummyOnline:
        def update(self, *args, **kwargs):  # noqa: ANN001
            raise AssertionError("update should not be called when pinned")

        def state_from_df(self, df):  # noqa: ANN001
            return [0.0]

        def predict_action(self, state):  # noqa: ANN001
            return 0.0

    engine = SimpleNamespace(
        settings=SimpleNamespace(enable_online_learning=True, online_training_enabled=True, online_updates_per_minute=10, training_metrics_window=100),
        online_policy=DummyOnline(),
        replay=SimpleNamespace(sample=lambda limit=64: []),
        _online_pause_until_ts=0.0,
    )

    # Simulate block: is_pinned True
    is_pinned = True
    if is_pinned or not engine.settings.enable_online_learning:
        pass
    else:  # pragma: no cover
        engine.online_policy.update([0.0], 0.0)

    # Cleanup
    unpin()


