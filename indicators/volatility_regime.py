import numpy as np
import pandas as pd
from .base import Indicator, IndicatorResult

class VolatilityRegime(Indicator):
    """Measures current volatility relative to recent history."""

    def __init__(self, period: int = 30):
        self.period = period

    def compute(self, ohlc: pd.DataFrame) -> IndicatorResult:
        df = ohlc.copy()
        
        # Daily return volatility
        ret = df["Close"].pct_change()
        hist_vol = ret.rolling(self.period).std() * np.sqrt(252)
        
        # Rank current vol against a long baseline (e.g., 252 days) to find regime
        # If we don't have 252 days, just rank against available
        long_period = min(252, len(df)) if len(df) > self.period else self.period
        if long_period <= self.period:
            return IndicatorResult(values=pd.Series(0.5, index=df.index), name=self.name)
            
        long_vol = ret.rolling(long_period).std() * np.sqrt(252)
        
        # Regime score: 0.0 (lowest vol) to 1.0 (highest vol)
        # We approximate by comparing short vol to long vol
        ratio = hist_vol / long_vol.replace(0, np.nan)
        
        # Cap ratio between 0.5 and 2.0, map to 0-1
        ratio_capped = ratio.clip(0.5, 2.0)
        regime = (ratio_capped - 0.5) / 1.5
        
        return IndicatorResult(values=regime.fillna(0.5), raw=df, name=self.name)

    @property
    def name(self) -> str:
        return f"VolRegime({self.period})"
