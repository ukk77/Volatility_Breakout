import sys
sys.path.insert(0, r"c:\Users\ukard\OneDrive\Desktop\trading")
sys.path.insert(0, r"c:\Users\ukard\OneDrive\Desktop\trading\risk_calculator\backend")

from app.services.market_data import fetch_ohlcv
from volatility_breakout.config import VolatilityBreakoutConfig
from volatility_breakout.signals.generator import compute_indicators

cfg = VolatilityBreakoutConfig()
ohlc = fetch_ohlcv("NVDA", 60)
df = compute_indicators(ohlc, cfg)

print(df.tail()[["Close", "Volume", "vol_sma", "squeeze_on", "bb_upper", "kc_upper", "adx", "vol_regime"]])
