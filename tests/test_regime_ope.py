import numpy as np
import pandas as pd

from nge_trader.ai.regime import RegimeDetector
from nge_trader.ai.ope import off_policy_evaluation


def test_regime_detector_runs():
    df = pd.DataFrame({"close": np.linspace(100, 110, 200) + np.random.randn(200)})
    rd = RegimeDetector(lookback=30, k=2)
    rd.fit(df)
    r = rd.predict(df)
    assert r in (0, 1)


def test_ope_runs():
    r = pd.Series(np.random.randn(200) * 0.001)
    b = pd.Series(np.random.choice([-1.0, 0.0, 1.0], size=200))
    c = pd.Series(np.random.choice([-1.0, 0.0, 1.0], size=200))
    out = off_policy_evaluation(r, b, c)
    assert "weighted_mean" in out and "weighted_sharpe" in out


