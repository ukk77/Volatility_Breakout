from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class SqueezeConfig:
    """TTM Squeeze configuration: Bollinger Bands strictly inside Keltner Channels."""
    enabled: bool = True
    length: int = 20
    bb_std: float = 2.0
    kc_mult: float = 1.5


@dataclass
class BreakoutConfig:
    """Breakout parameters."""
    enabled: bool = True
    volume_mult: float = 1.5
    volume_length: int = 20
    vwap_filter: bool = True          # P2: VWAP Confirmation (rolling MVWAP)
    expansion_confirm: bool = True    # P2: Must close outside Keltner Channel
    use_donchian: bool = False        # P3: Price must close above Donchian 20-day high (opt-in)
    donchian_period: int = 20         # Lookback for Donchian channel
    use_anchored_vwap: bool = False   # P3: Price must close above VWAP anchored to squeeze start (opt-in)


@dataclass
class ExitConfig:
    """Exit parameters."""
    trailing_ema_length: int = 8  # Close when price closes below this EMA


@dataclass
class ADXConfig:
    """Trend strength filter."""
    enabled: bool = True
    period: int = 14
    min_adx: float = 20.0  # Only take breakouts if trend is strong enough

@dataclass
class VolatilityRegimeConfig:
    """Volatility regime — scale down position size in high-vol periods."""
    enabled: bool = True
    period: int = 30
    low_vol_threshold: float = 0.15
    high_vol_threshold: float = 0.30
    min_multiplier: float = 0.25

@dataclass
class SignalConfig:
    """Signal generation settings."""
    sentiment_filter_enabled: bool = True
    min_sentiment_confidence: float = 0.4
    block_on_negative_sentiment: bool = True
    risk_filter_enabled: bool = True
    max_risk_score: float = 75.0

@dataclass
class PortfolioConstraintsConfig:
    max_open_positions: int = 10
    max_sector_exposure_pct: float = 40.0
    max_gross_exposure_pct: float = 100.0
    adv_participation_pct: float = 2.5

@dataclass
class PositionSizingConfig:
    """Position sizing parameters."""
    base_position_pct: float = 80.0
    max_position_pct: float = 100.0


@dataclass
class BacktestConfig:
    """Backtest engine settings."""
    initial_capital: float = 100_000.0
    commission_per_trade: float = 0.0
    commission_pct: float = 0.001
    slippage: float = 0.0005
    benchmark_ticker: str = "SPY"
    abs_return_hurdle: float = 0.03
    model_cash_interest: bool = True


@dataclass
class VolatilityBreakoutConfig:
    """Master configuration."""
    squeeze: SqueezeConfig = field(default_factory=SqueezeConfig)
    breakout: BreakoutConfig = field(default_factory=BreakoutConfig)
    exits: ExitConfig = field(default_factory=ExitConfig)
    adx: ADXConfig = field(default_factory=ADXConfig)
    vol_regime: VolatilityRegimeConfig = field(default_factory=VolatilityRegimeConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    portfolio_constraints: PortfolioConstraintsConfig = field(default_factory=PortfolioConstraintsConfig)
    position_sizing: PositionSizingConfig = field(default_factory=PositionSizingConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)

    sector_map: Dict[str, str] = field(default_factory=lambda: {
        "EQT": "Energy",
        "GEV": "Industrials",
        "VST": "Utilities",
        "NVDA": "Technology", "AMD": "Technology", "META": "Technology", "NFLX": "Technology",
        "AVGO": "Technology", "QQQ": "Technology", "TSLA": "Consumer Discretionary",
        "COIN": "Financials", "MSTR": "Technology", "PLTR": "Technology", "UBER": "Industrials",
        "SPY": "Diversified",
        "MU": "Technology", "LITE": "Technology", "NVTS": "Technology", "ASML": "Technology",
        "FCX": "Materials", "GE": "Industrials", "LMT": "Industrials", "RTX": "Industrials",
        "NUE": "Materials", "SMCI": "Technology", "MARA": "Financials"
    })

    tickers: List[str] = field(default_factory=lambda: [
        "NVDA", "TSLA", "AMD", "META", "NFLX",
        "COIN", "MSTR", "PLTR", "UBER", "AVGO",
        "QQQ", "SPY",
        "MU", "LITE", "NVTS", "ASML",
        "FCX", "GE", "LMT", "RTX", "NUE",
        "SMCI", "MARA"
    ])
    
    lookback_days: int = 7300


DEFAULT_CONFIG = VolatilityBreakoutConfig()
