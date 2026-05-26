from typing import Optional
from dataclasses import dataclass
from ..signals.generator import Signal
from ..config import VolatilityBreakoutConfig

def compute_position_dollars(
    sig: Signal,
    portfolio_nav: float,
    cfg: VolatilityBreakoutConfig,
    kelly_fraction: Optional[float] = None
) -> float:
    """Calculate the target dollar amount for a new position."""
    if portfolio_nav <= 0:
        return 0.0
        
    base_pct = cfg.position_sizing.base_position_pct / 100.0
    max_pct = cfg.position_sizing.max_position_pct / 100.0
    
    # 1. Base allocation
    alloc_pct = base_pct
    
    # 2. Risk/Kelly adjustment
    if kelly_fraction is not None and kelly_fraction > 0:
        # Scale base up or down based on Kelly (very simplistically here)
        alloc_pct = min(kelly_fraction, max_pct)
        
    # 3. Sentiment agreement multiplier
    if sig.sentiment is not None:
        if sig.sentiment == "positive":
            pass # We only go LONG in this strategy, so positive is agreement
        elif sig.sentiment == "negative":
            alloc_pct *= 0.5
            
    # Ensure bounds
    alloc_pct = min(alloc_pct, max_pct)
    return portfolio_nav * alloc_pct

def shares_to_buy(
    sig: Signal,
    portfolio_nav: float,
    current_price: float,
    cfg: VolatilityBreakoutConfig,
    kelly_fraction: Optional[float] = None,
    daily_volume: Optional[float] = None
) -> int:
    """Calculate number of shares to buy."""
    if current_price <= 0:
        return 0
        
    dollars = compute_position_dollars(sig, portfolio_nav, cfg, kelly_fraction)
    shares = int(dollars / current_price)
    
    # ADV Participation cap
    if cfg.portfolio_constraints.adv_participation_pct > 0 and daily_volume is not None and daily_volume > 0:
        max_shares_by_adv = int(daily_volume * (cfg.portfolio_constraints.adv_participation_pct / 100.0))
        shares = min(shares, max_shares_by_adv)
        
    return max(0, shares)
