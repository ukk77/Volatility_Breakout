import numpy as np
import pandas as pd
from .base import Indicator, IndicatorResult

class ADX(Indicator):
    """Average Directional Index (ADX)"""

    def __init__(self, period: int = 14):
        self.period = period

    def compute(self, ohlc: pd.DataFrame) -> IndicatorResult:
        df = ohlc.copy()
        
        df["up_move"] = df["High"] - df["High"].shift(1)
        df["down_move"] = df["Low"].shift(1) - df["Low"]
        
        df["+dm"] = np.where((df["up_move"] > df["down_move"]) & (df["up_move"] > 0), df["up_move"], 0)
        df["-dm"] = np.where((df["down_move"] > df["up_move"]) & (df["down_move"] > 0), df["down_move"], 0)
        
        df["tr0"] = abs(df["High"] - df["Low"])
        df["tr1"] = abs(df["High"] - df["Close"].shift(1))
        df["tr2"] = abs(df["Low"] - df["Close"].shift(1))
        df["tr"] = df[["tr0", "tr1", "tr2"]].max(axis=1)
        
        tr_sm = df["tr"].rolling(self.period).sum()
        pdm_sm = df["+dm"].rolling(self.period).sum()
        ndm_sm = df["-dm"].rolling(self.period).sum()
        
        # Smoothed true range might have zeros in edge cases
        pdi = 100 * (pdm_sm / tr_sm.replace(0, np.nan))
        ndi = 100 * (ndm_sm / tr_sm.replace(0, np.nan))
        
        dx = 100 * (abs(pdi - ndi) / (pdi + ndi).replace(0, np.nan))
        adx = dx.rolling(self.period).mean()
        
        return IndicatorResult(values=adx.fillna(0), raw=df, name=self.name)

    @property
    def name(self) -> str:
        return f"ADX({self.period})"
