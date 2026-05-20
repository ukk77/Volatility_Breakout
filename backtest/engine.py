import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from ..config import VolatilityBreakoutConfig
from ..signals.generator import compute_indicators
from .portfolio import Portfolio
from .metrics import compute_all_metrics


@dataclass
class BacktestResult:
    ticker: str
    equity_curve: pd.Series
    trades_df: pd.DataFrame
    metrics: Dict
    benchmark_equity: Dict[str, pd.Series]


@dataclass
class BacktestSummary:
    results: Dict[str, BacktestResult] = field(default_factory=dict)
    portfolio_equity: Optional[pd.Series] = None
    portfolio_metrics: Optional[Dict] = None


def _cash_hurdle_equity(
    equity: pd.Series,
    initial_capital: float,
    rf_annual: float,
    hurdle: float = 0.03,
) -> pd.Series:
    n = len(equity)
    if n == 0:
        return pd.Series(dtype=float)
    daily_factor = (1.0 + rf_annual + hurdle) ** (1.0 / 252)
    vals = initial_capital * (daily_factor ** np.arange(1, n + 1))
    return pd.Series(vals.astype(float), index=equity.index)


def run_backtest(
    cfg: VolatilityBreakoutConfig,
    ticker_ohlc: Dict[str, pd.DataFrame],
    benchmark_ohlc: Dict[str, pd.DataFrame],
    rf_annual: float = 0.04,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> BacktestSummary:
    summary = BacktestSummary()
    
    # Run each ticker independently
    for ticker, ohlc in ticker_ohlc.items():
        df = ohlc.copy()
        if start_date:
            df = df[df.index >= pd.Timestamp(start_date)]
        if end_date:
            df = df[df.index <= pd.Timestamp(end_date)]
            
        if len(df) < 50:
            continue
            
        df = compute_indicators(df, cfg)
        portfolio = Portfolio(cfg.backtest.initial_capital)
        
        position = 0
        entry_price = 0.0
        stop_loss = 0.0
        shares = 0
        
        for i in range(1, len(df)):
            curr = df.iloc[i]
            prev = df.iloc[i-1]
            date_str = curr.name.strftime("%Y-%m-%d") if isinstance(curr.name, pd.Timestamp) else str(curr.name)
            current_price = float(curr["Close"])
            
            # Accrue interest
            if cfg.backtest.model_cash_interest:
                daily_rate = rf_annual / 252.0
                portfolio.cash += portfolio.cash * daily_rate
            
            if position == 0:
                # Look for entry
                squeeze_fired = bool(prev["squeeze_on"]) and not bool(curr["squeeze_on"])
                breakout_up = current_price > float(curr["bb_upper"])
                vol_surge = float(curr["Volume"]) > (float(curr["vol_sma"]) * cfg.breakout.volume_mult)
                
                if squeeze_fired and breakout_up and vol_surge:
                    # Buy!
                    shares_to_buy = int((portfolio.cash * (cfg.position_sizing.base_position_pct / 100.0)) / current_price)
                    if portfolio.buy(ticker, shares_to_buy, current_price, date_str):
                        position = 1
                        stop_loss = float(curr["Low"])  # Stop at low of breakout candle
            elif position == 1:
                shares = portfolio.shares_held(ticker)
                # Look for exit
                # 1. Stop loss hit
                if float(curr["Low"]) <= stop_loss:
                    exit_price = min(current_price, stop_loss)  # Slippage simplified
                    portfolio.sell(ticker, shares, exit_price, date_str)
                    position = 0
                # 2. Trailing EMA exit
                elif current_price < float(curr["ema_exit"]):
                    portfolio.sell(ticker, shares, current_price, date_str)
                    position = 0
                else:
                    pass
                    
            # Record equity
            portfolio.record_equity(date_str, {ticker: current_price} if position == 1 else {})
            
        # End of ticker loop
        if position == 1:
            shares = portfolio.shares_held(ticker)
            portfolio.sell(ticker, shares, current_price, date_str)
            
        equity = portfolio.equity_series()
        trades = portfolio.to_trades_df()
        
        bench_equities: Dict[str, pd.Series] = {}
        for b_name, b_ohlc in benchmark_ohlc.items():
            b_f = b_ohlc.copy()
            if start_date:
                b_f = b_f[b_f.index >= pd.Timestamp(start_date)]
            if end_date:
                b_f = b_f[b_f.index <= pd.Timestamp(end_date)]
            b_close = b_f["Close"].dropna()
            if not b_close.empty:
                bench_equities[b_name] = cfg.backtest.initial_capital * (b_close / b_close.iloc[0])
                
        bench_equities["cash_plus_3"] = _cash_hurdle_equity(
            equity, cfg.backtest.initial_capital, rf_annual, cfg.backtest.abs_return_hurdle
        )
        
        metrics = compute_all_metrics(equity, cfg.backtest.initial_capital, trades, bench_equities, rf_annual)
        
        summary.results[ticker] = BacktestResult(
            ticker=ticker,
            equity_curve=equity,
            trades_df=trades,
            metrics=metrics,
            benchmark_equity=bench_equities
        )
        
    # Aggregate portfolio
    valid_curves = [r.equity_curve for r in summary.results.values() if not r.equity_curve.empty]
    if valid_curves:
        normalised = [c / c.iloc[0] for c in valid_curves]
        combined_norm = pd.concat(normalised, axis=1).ffill().mean(axis=1)
        combined_equity = cfg.backtest.initial_capital * combined_norm
        summary.portfolio_equity = combined_equity
        
        bench_equities = {}
        for b_name, b_ohlc in benchmark_ohlc.items():
            b_f = b_ohlc.copy()
            if start_date:
                b_f = b_f[b_f.index >= pd.Timestamp(start_date)]
            if end_date:
                b_f = b_f[b_f.index <= pd.Timestamp(end_date)]
            b_close = b_f["Close"].dropna()
            if not b_close.empty:
                bench_equities[b_name] = cfg.backtest.initial_capital * (b_close / b_close.iloc[0])
                
        bench_equities["cash_plus_3"] = _cash_hurdle_equity(
            combined_equity, cfg.backtest.initial_capital, rf_annual, cfg.backtest.abs_return_hurdle
        )
        
        all_trades = [r.trades_df for r in summary.results.values() if not r.trades_df.empty]
        combined_trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
        
        summary.portfolio_metrics = compute_all_metrics(
            combined_equity, cfg.backtest.initial_capital, combined_trades, bench_equities, rf_annual
        )
        
    return summary
