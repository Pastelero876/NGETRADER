from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Strategy(ABC):
    """Interfaz base para estrategias de trading."""

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:  # type: ignore[name-defined]
        """Genera señales a partir de un DataFrame de precios."""

        raise NotImplementedError


