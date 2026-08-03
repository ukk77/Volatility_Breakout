import pandas as pd
import numpy as np
import os
import requests
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict
from enum import Enum
from ..config import VolatilityBreakoutConfig
from ..indicators import ADX, VolatilityRegime
from .filters import apply_vb_filters
from trading_core.signal_strength import saturate

try:
    from trading_core.session_context import (
        fetch_premarket_gap,
        premarket_confirmation_mult,
        early_session_size_scalar,
    )
except ImportError:
    def fetch_premarket_gap(ticker): return None  # type: ignore
    def premarket_confirmation_mult(gap, direction, **kw): return 1.0  # type: ignore
    def early_session_size_scalar(**kw): return 1.0  # type: ignore

import logging
log = logging.getLogger(__name__)

def _fetch_latest_sentiment(ticker: str) -> Optional[dict]:
    url = os.getenv("SENTIMENT_API_URL", "http://localhost:8000")
    try:
        resp = requests.get(f"{url}/api/history/{ticker}?limit=1", timeout=3)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("snapshots") and len(data["snapshots"]) > 0:
                return data["snapshots"][0]
        else:
            log.warning("fetch_sentiment %s returned status %s", ticker, resp.status_code)
    except Exception as e:
        log.warning("fetch_sentiment %s failed: %s", ticker, type(e).__name__)
    return None

def _fetch_latest_risk(ticker: str) -> Optional[dict]:
    url = os.getenv("RISK_API_URL", "http://localhost:8100")
    try:
        resp = requests.get(f"{url}/api/history/{ticker}?limit=1", timeout=3)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("snapshots") and len(data["snapshots"]) > 0:
                return data["snapshots"][0]
        else:
            log.warning("fetch_risk %s returned status %s", ticker, resp.status_code)
    except Exception as e:
        log.warning("fetch_risk %s failed: %s", ticker, type(e).__name__)
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
    filtered_strength: float = 0.0
    raw_strength: float = 0.0
    sentiment: Optional[str] = None
    kelly_fraction: Optional[float] = None
    adx_value: Optional[float] = None
    volume_ratio: Optional[float] = None
    vol_regime_mult: Optional[float] = None


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

    # P2: VWAP Confirmation (20-period MVWAP)
    df["typical_price"] = (df["High"] + df["Low"] + df["Close"]) / 3
    df["mvwap"] = (df["typical_price"] * df["Volume"]).rolling(length).sum() / df["Volume"].rolling(length).sum()

    # P3: Donchian Channel — rolling high/low
    dc_period = cfg.breakout.donchian_period
    df["donchian_high"] = df["High"].rolling(dc_period).max()
    df["donchian_low"] = df["Low"].rolling(dc_period).min()

    # P3: Anchored VWAP — VWAP reset to start of each squeeze episode
    squeeze_starts = df["squeeze_on"] & (~df["squeeze_on"].shift(1).fillna(False))
    df["squeeze_episode"] = squeeze_starts.cumsum()
    df.loc[~df["squeeze_on"], "squeeze_episode"] = np.nan
    anchored_vwap = pd.Series(np.nan, index=df.index)
    for ep_id, grp in df.dropna(subset=["squeeze_episode"]).groupby("squeeze_episode"):
        cum_tpv = (grp["typical_price"] * grp["Volume"]).cumsum()
        cum_vol = grp["Volume"].cumsum().replace(0, np.nan)
        anchored_vwap.loc[grp.index] = cum_tpv / cum_vol
    df["anchored_vwap"] = anchored_vwap.ffill()

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

def generate_signal(ticker: str, ohlc: pd.DataFrame, cfg: VolatilityBreakoutConfig, sentiment_override: Optional[float] = None, sentiment_data: Optional[Dict] = None, risk_data: Optional[Dict] = None, precomputed_df: Optional[pd.DataFrame] = None) -> Signal:
    if len(ohlc) < max(cfg.squeeze.length, cfg.breakout.volume_length) + 1:
        return Signal(ticker, "", Action.HOLD, 0.0, 0.0, "Not enough data")

    df = precomputed_df if precomputed_df is not None else compute_indicators(ohlc, cfg)
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    date_str = last.name.strftime("%Y-%m-%d") if isinstance(last.name, pd.Timestamp) else str(last.name)
    price = float(last["Close"])
    
    # Are we firing a breakout today?
    # 1. We must have been in a squeeze yesterday (or today)
    was_squeezed = bool(prev["squeeze_on"]) or bool(last["squeeze_on"])
    
    # 2. Price closes above Upper BB
    breakout_up = float(last["Close"]) > float(last["bb_upper"])
    
    # P2: Expansion confirmation (must close outside Keltner)
    if getattr(cfg.breakout, 'expansion_confirm', False):
        breakout_up = breakout_up and (float(last["Close"]) > float(last["kc_upper"]))

    # P2: VWAP confirmation (must close above institutional average)
    if getattr(cfg.breakout, 'vwap_filter', False):
        breakout_up = breakout_up and (float(last["Close"]) > float(last["mvwap"]))

    # P3: Donchian Channel — must close at or above 20-day high
    if getattr(cfg.breakout, 'use_donchian', False) and pd.notna(last["donchian_high"]):
        breakout_up = breakout_up and (float(last["Close"]) >= float(last["donchian_high"]))

    # P3: Anchored VWAP — must close above VWAP anchored to squeeze start
    if getattr(cfg.breakout, 'use_anchored_vwap', False) and pd.notna(last["anchored_vwap"]):
        breakout_up = breakout_up and (float(last["Close"]) > float(last["anchored_vwap"]))
    
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

        pm_gap = fetch_premarket_gap(ticker)
        pm_mult = premarket_confirmation_mult(pm_gap, "BUY")
        es_scalar = early_session_size_scalar()
        session_notes = []
        if pm_gap is not None and abs(pm_gap) >= 0.005:
            session_notes.append(f"pm_gap={pm_gap:+.1%}(x{pm_mult:.2f})")
        if es_scalar < 1.0:
            session_notes.append(f"early_session(x{es_scalar:.2f})")
        if kelly_val is not None:
            kelly_val = round(kelly_val * pm_mult * es_scalar, 4)
        reason_str = f"Squeeze breakout up on {last['Volume']/last['vol_sma']:.1f}x volume | {meta_str}"
        if session_notes:
            reason_str += " | " + " | ".join(session_notes)

        # ── Compute signal strength ────────────────────────────────────────────
        # Base: volume surge ratio normalised (1.5x=0.5, 3x=1.0)
        vol_surge_ratio = float(last["Volume"]) / float(last["vol_sma"])
        vol_component = min((vol_surge_ratio - 1.0) / 2.0, 1.0)  # 0.0 at 1x, 1.0 at 3x

        # ADX component: stronger trend = better breakout (0.5 at min_adx, 1.0 at 2*min_adx)
        adx_component = min(adx_val / (2.0 * cfg.adx.min_adx), 1.0) if (cfg.adx.enabled and adx_val is not None) else 0.7

        # Sentiment multiplier
        sent_mult = 1.0
        if sent_str == "positive":
            sent_mult = cfg.position_sizing.sentiment_agree_mult
        elif sent_str == "negative":
            sent_mult = cfg.position_sizing.sentiment_disagree_mult
        else:
            sent_mult = cfg.position_sizing.sentiment_neutral_mult

        # Vol regime multiplier
        vr_mult = 1.0
        if cfg.vol_regime.enabled and pd.notna(last.get("vol_regime", np.nan)):
            vr_mult = float(last["vol_regime"]) if float(last["vol_regime"]) > 0 else 1.0

        raw_str = vol_component * adx_component * sent_mult * vr_mult * pm_mult * es_scalar
        strength = saturate(raw_str)

        return Signal(
            ticker=ticker,
            date=date_str,
            action=Action.BUY,
            price=price,
            stop_loss=stop_loss,
            reason=reason_str,
            filtered_strength=strength,
            raw_strength=raw_str,
            sentiment=sent_str,
            kelly_fraction=kelly_val,
            adx_value=adx_val,
            volume_ratio=vol_surge_ratio,
            vol_regime_mult=vr_mult,
        )
        
    return Signal(
        ticker=ticker,
        date=date_str,
        action=Action.HOLD,
        price=price,
        stop_loss=0.0,
        reason="No setup"
    )
