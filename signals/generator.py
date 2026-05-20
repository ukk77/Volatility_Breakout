import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional
from enum import Enum
from ..config import VolatilityBreakoutConfig


class Action(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class Signal:
    ticker: str
    date: str
    action: Action
    price: float
    stop_loss: float
    reason: str


def compute_indicators(ohlc: pd.DataFrame, cfg: VolatilityBreakoutConfig) -> pd.DataFrame:
    df = ohlc.copy()
    
    # Volume SMA
    df["vol_sma"] = df["Volume"].rolling(cfg.breakout.volume_length).mean()
    
    # Squeeze (BB and KC)
    length = cfg.squeeze.length
    df["sma"] = df["Close"].rolling(length).mean()
    
    # BB
    std = df["Close"].rolling(length).std(ddof=0)
    df["bb_upper"] = df["sma"] + (cfg.squeeze.bb_std * std)
    df["bb_lower"] = df["sma"] - (cfg.squeeze.bb_std * std)
    
    # KC (True Range)
    df["tr0"] = abs(df["High"] - df["Low"])
    df["tr1"] = abs(df["High"] - df["Close"].shift())
    df["tr2"] = abs(df["Low"] - df["Close"].shift())
    df["tr"] = df[["tr0", "tr1", "tr2"]].max(axis=1)
    df["atr"] = df["tr"].rolling(length).mean()
    
    df["kc_upper"] = df["sma"] + (cfg.squeeze.kc_mult * df["atr"])
    df["kc_lower"] = df["sma"] - (cfg.squeeze.kc_mult * df["atr"])
    
    # Squeeze ON: BB is completely inside KC
    df["squeeze_on"] = (df["bb_lower"] > df["kc_lower"]) & (df["bb_upper"] < df["kc_upper"])
    
    # Fast EMA for exits
    df["ema_exit"] = df["Close"].ewm(span=cfg.exits.trailing_ema_length, adjust=False).mean()
    
    return df


def generate_signal(ticker: str, ohlc: pd.DataFrame, cfg: VolatilityBreakoutConfig) -> Signal:
    if len(ohlc) < max(cfg.squeeze.length, cfg.breakout.volume_length) + 1:
        return Signal(ticker, "", Action.HOLD, 0.0, 0.0, "Not enough data")
        
    df = compute_indicators(ohlc, cfg)
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    date_str = last.name.strftime("%Y-%m-%d") if isinstance(last.name, pd.Timestamp) else str(last.name)
    price = float(last["Close"])
    
    # Are we firing a breakout today?
    # 1. We must have been in a squeeze yesterday (or today)
    was_squeezed = bool(prev["squeeze_on"]) or bool(last["squeeze_on"])
    
    # 2. Price closes above Upper BB
    breakout_up = float(last["Close"]) > float(last["bb_upper"])
    
    # 3. Volume surge
    vol_surge = float(last["Volume"]) > (float(last["vol_sma"]) * cfg.breakout.volume_mult)
    
    if was_squeezed and breakout_up and vol_surge:
        # Buy! Stop loss is the low of the breakout candle
        stop_loss = float(last["Low"])
        return Signal(
            ticker=ticker,
            date=date_str,
            action=Action.BUY,
            price=price,
            stop_loss=stop_loss,
            reason=f"Squeeze breakout up on {last['Volume']/last['vol_sma']:.1f}x volume"
        )
        
    return Signal(
        ticker=ticker,
        date=date_str,
        action=Action.HOLD,
        price=price,
        stop_loss=0.0,
        reason="No setup"
    )
