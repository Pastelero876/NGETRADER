from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Type

from .base import Strategy
from .moving_average import MovingAverageCrossStrategy
from .rsi import RSIStrategy
from .macd import MACDStrategy
from .bollinger import BollingerBandsStrategy
from .stochastic import StochasticOscillatorStrategy
from .donchian import DonchianBreakoutStrategy
from .momentum import MomentumStrategy
from .cci import CCIStrategy
from .adx import ADXStrategy
from .keltner import KeltnerChannelsStrategy
from .williamsr import WilliamsRStrategy
from .psar import ParabolicSARStrategy
from .ichimoku import IchimokuStrategy
from .supertrend import SupertrendStrategy
from .obv import OBVStrategy
from .ema_crossover import EMACrossoverStrategy
from .pivot import PivotReversalStrategy
from .heikin_ashi import HeikinAshiTrendStrategy
from .trix import TRIXStrategy
from .roc import ROCStrategy
from .zscore import ZScoreReversionStrategy
from .cmf import ChaikinMoneyFlowStrategy
from .vwap import VWAPDeviationStrategy
from .pairs import PairsTradingStrategy


@dataclass
class StrategyInfo:
    key: str
    name: str
    cls: Type[Strategy]


STRATEGY_REGISTRY: Dict[str, StrategyInfo] = {
    "ma_cross": StrategyInfo(key="ma_cross", name="Cruce de Medias", cls=MovingAverageCrossStrategy),
    "rsi": StrategyInfo(key="rsi", name="RSI", cls=RSIStrategy),
    "macd": StrategyInfo(key="macd", name="MACD", cls=MACDStrategy),
    "bollinger": StrategyInfo(key="bollinger", name="Bollinger Bands", cls=BollingerBandsStrategy),
    "stochastic": StrategyInfo(key="stochastic", name="Stochastic", cls=StochasticOscillatorStrategy),
    "donchian": StrategyInfo(key="donchian", name="Donchian Breakout", cls=DonchianBreakoutStrategy),
    "momentum": StrategyInfo(key="momentum", name="Momentum", cls=MomentumStrategy),
    "cci": StrategyInfo(key="cci", name="CCI", cls=CCIStrategy),
    "adx": StrategyInfo(key="adx", name="ADX", cls=ADXStrategy),
    "keltner": StrategyInfo(key="keltner", name="Keltner Channels", cls=KeltnerChannelsStrategy),
    "williamsr": StrategyInfo(key="williamsr", name="Williams %R", cls=WilliamsRStrategy),
    "psar": StrategyInfo(key="psar", name="Parabolic SAR", cls=ParabolicSARStrategy),
    "ichimoku": StrategyInfo(key="ichimoku", name="Ichimoku", cls=IchimokuStrategy),
    "supertrend": StrategyInfo(key="supertrend", name="Supertrend", cls=SupertrendStrategy),
    "obv": StrategyInfo(key="obv", name="OBV", cls=OBVStrategy),
    "ema_x": StrategyInfo(key="ema_x", name="EMA Crossover", cls=EMACrossoverStrategy),
    "pivot": StrategyInfo(key="pivot", name="Pivot Reversal", cls=PivotReversalStrategy),
    "heikin": StrategyInfo(key="heikin", name="Heikin Ashi Trend", cls=HeikinAshiTrendStrategy),
    "trix": StrategyInfo(key="trix", name="TRIX", cls=TRIXStrategy),
    "roc": StrategyInfo(key="roc", name="Rate of Change", cls=ROCStrategy),
    "zscore": StrategyInfo(key="zscore", name="Z-Score Reversion", cls=ZScoreReversionStrategy),
    "cmf": StrategyInfo(key="cmf", name="Chaikin Money Flow", cls=ChaikinMoneyFlowStrategy),
    "vwap_dev": StrategyInfo(key="vwap_dev", name="VWAP Deviation", cls=VWAPDeviationStrategy),
    "pairs": StrategyInfo(key="pairs", name="Pairs Trading", cls=PairsTradingStrategy),
}


