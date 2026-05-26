import pandas as pd
import numpy as np
import sqlite3
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict
from enum import Enum
from ..config import VolatilityBreakoutConfig
from ..indicators import ADX, VolatilityRegime
from .filters import apply_vb_filters

_TRADING_ROOT = Path(__file__).resolve().parents[3]
_SENTIMENT_DB = _TRADING_ROOT / "sentiment_analysis" / "backend" / "sentiment_history.db"
_RISK_DB = _TRADING_ROOT / "risk_calculator" / "backend" / "risk_history.db"

def _fetch_latest_sentiment(ticker: str) -> Optional[dict]:
    if not _SENTIMENT_DB.exists():
        return None
    try:
        with sqlite3.connect(str(_SENTIMENT_DB)) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT overall_sentiment, confidence, avg_sentiment "
                "FROM sentiment_snapshots "
                "WHERE UPPER(ticker)=UPPER(?) ORDER BY captured_at DESC LIMIT 1",
                (ticker.upper(),),
            ).fetchone()
        return dict(row) if row else None
    except Exception:
        return None

def _fetch_latest_risk(ticker: str) -> Optional[dict]:
    if not _RISK_DB.exists():
        return None
    try:
        with sqlite3.connect(str(_RISK_DB)) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT composite_risk_score, risk_bucket, overall_sentiment, upstream_confidence, kelly_fraction_capped, suggested_stop_loss_pct "
                "FROM risk_snapshots "
                "WHERE UPPER(ticker)=UPPER(?) ORDER BY captured_at DESC LIMIT 1",
                (ticker.upper(),),
            ).fetchone()
        return dict(row) if row else None
    except Exception:
        return None



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
    sentiment: Optional[str] = None
    kelly_fraction: Optional[float] = None


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

    # Trend Strength (ADX)
    if cfg.adx.enabled:
        adx_ind = ADX(period=cfg.adx.period)
        df["adx"] = adx_ind.compute(df).values
    else:
        df["adx"] = np.nan

    # Volatility Regime
    if cfg.vol_regime.enabled:
        vr_ind = VolatilityRegime(period=cfg.vol_regime.period)
        df["vol_regime"] = vr_ind.compute(df).values
    else:
        df["vol_regime"] = np.nan
        
    return df

def generate_signal(ticker: str, ohlc: pd.DataFrame, cfg: VolatilityBreakoutConfig, sentiment_override: Optional[float] = None, sentiment_data: Optional[Dict] = None, risk_data: Optional[Dict] = None) -> Signal:
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
        adx_val = float(last["adx"]) if pd.notna(last["adx"]) else None
        if sentiment_data is None and sentiment_override is None: sentiment_data = _fetch_latest_sentiment(ticker)
        if risk_data is None: risk_data = _fetch_latest_risk(ticker)
        passed, block_reason, meta = apply_vb_filters(ticker, sentiment_override, adx_val, cfg, sentiment_data, risk_data)
        
        if not passed:
            return Signal(
                ticker=ticker,
                date=date_str,
                action=Action.HOLD,
                price=price,
                stop_loss=0.0,
                reason=f"Blocked: {block_reason}"
            )
            
        # Buy! Stop loss is the low of the breakout candle
        stop_loss = float(last["Low"])
        meta_str = " | ".join(meta.values())
        
        sent_str = "neutral"
        if sentiment_override is not None:
            sent_str = "positive" if sentiment_override > 0 else ("negative" if sentiment_override < 0 else "neutral")
        elif sentiment_data is not None:
            sent_str = sentiment_data.get("overall_sentiment", "neutral")

        kelly_val = float(risk_data.get("kelly_fraction_capped", 0.0)) if risk_data else None

        return Signal(
            ticker=ticker,
            date=date_str,
            action=Action.BUY,
            price=price,
            stop_loss=stop_loss,
            reason=f"Squeeze breakout up on {last['Volume']/last['vol_sma']:.1f}x volume | {meta_str}",
            sentiment=sent_str,
            kelly_fraction=kelly_val
        )
        
    return Signal(
        ticker=ticker,
        date=date_str,
        action=Action.HOLD,
        price=price,
        stop_loss=0.0,
        reason="No setup"
    )
