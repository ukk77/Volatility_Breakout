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


@dataclass
class ExitConfig:
    """Exit parameters."""
    trailing_ema_length: int = 8  # Close when price closes below this EMA


@dataclass
class PositionSizingConfig:
    """Position sizing parameters."""
    base_position_pct: float = 80.0
    max_position_pct: float = 100.0


@dataclass
class BacktestConfig:
    """Backtest engine settings."""
    initial_capital: float = 100_000.0
    benchmark_ticker: str = "SPY"
    abs_return_hurdle: float = 0.03
    model_cash_interest: bool = True


@dataclass
class VolatilityBreakoutConfig:
    """Master configuration."""
    squeeze: SqueezeConfig = field(default_factory=SqueezeConfig)
    breakout: BreakoutConfig = field(default_factory=BreakoutConfig)
    exits: ExitConfig = field(default_factory=ExitConfig)
    position_sizing: PositionSizingConfig = field(default_factory=PositionSizingConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)

    tickers: List[str] = field(default_factory=lambda: [
        "NVDA", "TSLA", "AMD", "META", "NFLX",
        "COIN", "MSTR", "PLTR", "UBER", "AVGO",
        "QQQ", "SPY"
    ])
    
    lookback_days: int = 7300


DEFAULT_CONFIG = VolatilityBreakoutConfig()
