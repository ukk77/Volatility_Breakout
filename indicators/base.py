import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

@dataclass
class IndicatorResult:
    values: pd.Series
    raw: Optional[pd.DataFrame] = None
    name: str = ""

class Indicator(ABC):
    @abstractmethod
    def compute(self, ohlc: pd.DataFrame) -> IndicatorResult:
        ...

    def signal_series(self, ohlc: pd.DataFrame) -> pd.Series:
        result = self.compute(ohlc)
        return result.values.map(lambda x: 1.0 if x > 0 else (-1.0 if x < 0 else 0.0))

    @property
    @abstractmethod
    def name(self) -> str:
        ...
