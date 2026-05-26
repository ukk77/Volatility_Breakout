from typing import Optional, Dict, Tuple
from ..config import VolatilityBreakoutConfig
import logging

log = logging.getLogger(__name__)

def apply_vb_filters(
    ticker: str,
    sentiment_override: Optional[float],
    adx_val: Optional[float],
    cfg: VolatilityBreakoutConfig,
    sentiment_data: Optional[Dict] = None,
    risk_data: Optional[Dict] = None
) -> Tuple[bool, str, Dict[str, str]]:
    """
    Apply Volatility Breakout filters (trend, sentiment, risk).
    Returns: (passed: bool, block_reason: str, meta: dict)
    """
    meta = {}

    # 1. ADX Filter (Trend confirmation)
    if cfg.adx.enabled and adx_val is not None:
        if adx_val < cfg.adx.min_adx:
            reason = f"adx={adx_val:.1f}<{cfg.adx.min_adx}(weak_trend)"
            meta["adx"] = reason
            return False, reason, meta
        meta["adx"] = f"adx={adx_val:.1f}>={cfg.adx.min_adx}(trend_OK)"

    # 2. Sentiment Filter
    if cfg.signal.sentiment_filter_enabled:
        if sentiment_override is not None:
            sentiment_score = sentiment_override
            if cfg.signal.block_on_negative_sentiment and sentiment_score < 0.0:
                reason = f"sent_score={sentiment_score:.2f}(negative)"
                meta["sent"] = reason
                return False, reason, meta
            meta["sent"] = f"sent={sentiment_score:.2f}OK"
        elif sentiment_data is not None:
            overall_sentiment = sentiment_data.get("overall_sentiment")
            conf = float(sentiment_data.get("confidence") or 0.0)
            
            if conf < cfg.signal.min_sentiment_confidence:
                reason = f"low_conf={conf:.2f}<{cfg.signal.min_sentiment_confidence}"
                meta["sent"] = reason
                return False, reason, meta
            elif cfg.signal.block_on_negative_sentiment and overall_sentiment == "negative":
                reason = "blocked:negative_sentiment"
                meta["sent"] = reason
                return False, reason, meta
            
            meta["sent"] = f"sent={overall_sentiment}({conf:.2f})OK"

    # 3. Risk Filter
    if cfg.signal.risk_filter_enabled:
        if risk_data is not None:
            risk_score = risk_data.get("composite_risk_score")
            if risk_score is not None:
                if risk_score > cfg.signal.max_risk_score:
                    reason = f"risk={risk_score:.1f}>{cfg.signal.max_risk_score}"
                    meta["risk"] = reason
                    return False, reason, meta
                meta["risk"] = f"risk={risk_score:.1f}OK"

    return True, "", meta
