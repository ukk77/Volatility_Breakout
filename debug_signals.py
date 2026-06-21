import os
import sys
from pathlib import Path

_TRADING_ROOT = Path(__file__).resolve().parents[1]
_RISK_BACKEND = Path(os.getenv("RISK_CALCULATOR_BACKEND", str(_TRADING_ROOT / "risk_calculator" / "backend")))
if str(_TRADING_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRADING_ROOT))
if str(_RISK_BACKEND) not in sys.path:
    sys.path.insert(0, str(_RISK_BACKEND))

from app.services.market_data import fetch_ohlcv
from volatility_breakout.config import VolatilityBreakoutConfig
from volatility_breakout.signals.generator import compute_indicators

cfg = VolatilityBreakoutConfig()
ohlc = fetch_ohlcv("NVDA", 60)
df = compute_indicators(ohlc, cfg)

print(df.tail()[["Close", "Volume", "vol_sma", "squeeze_on", "bb_upper", "kc_upper", "adx", "vol_regime"]])
