import os
import pandas as pd
import sqlite3
import sys
from pathlib import Path

_TRADING_ROOT = Path(__file__).resolve().parents[2]
_RISK_BACKEND = Path(os.getenv("RISK_CALCULATOR_BACKEND", str(_TRADING_ROOT / 'risk_calculator' / 'backend')))
if str(_RISK_BACKEND) not in sys.path:
    sys.path.insert(0, str(_RISK_BACKEND))
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from ..config import VolatilityBreakoutConfig
from ..signals.generator import compute_indicators, generate_signal, Action
from ..position_sizing.sizer import shares_to_buy
from trading_core import Portfolio
from trading_core import compute_all_metrics


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


def _load_sentiment_history(ticker: str) -> pd.DataFrame:
    db_path = _TRADING_ROOT / 'sentiment_analysis' / 'backend' / 'sentiment_history.db'
    if not db_path.exists():
        return pd.DataFrame()
    try:
        with sqlite3.connect(str(db_path)) as conn:
            df = pd.read_sql_query(
                'SELECT captured_at, overall_sentiment, confidence FROM sentiment_snapshots WHERE UPPER(ticker)=UPPER(?)',
                conn, params=(ticker.upper(),))
        if df.empty: return df
        df['date'] = pd.to_datetime(df['captured_at']).dt.date
        df = df.sort_values('date').drop_duplicates('date', keep='last')
        return df.set_index('date')
    except Exception:
        return pd.DataFrame()

def _load_risk_history(ticker: str) -> pd.DataFrame:
    db_path = _TRADING_ROOT / 'risk_calculator' / 'backend' / 'risk_history.db'
    if not db_path.exists():
        return pd.DataFrame()
    try:
        with sqlite3.connect(str(db_path)) as conn:
            df = pd.read_sql_query(
                'SELECT captured_at, composite_risk_score, risk_bucket, kelly_fraction_capped, suggested_stop_loss_pct FROM risk_snapshots WHERE UPPER(ticker)=UPPER(?)',
                conn, params=(ticker.upper(),))
        if df.empty: return df
        df['date'] = pd.to_datetime(df['captured_at']).dt.date
        df = df.sort_values('date').drop_duplicates('date', keep='last')
        return df.set_index('date')
    except Exception:
        return pd.DataFrame()

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
        sentiment_hist = _load_sentiment_history(ticker)
        risk_hist = _load_risk_history(ticker)
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
        bars_since_entry = 0
        entry_atr = 0.0  # ATR at entry for false-breakout detection
        
        for i in range(1, len(df)):
            curr = df.iloc[i]
            prev = df.iloc[i-1]
            date_str = curr.name.strftime("%Y-%m-%d") if isinstance(curr.name, pd.Timestamp) else str(curr.name)
            date_obj = pd.to_datetime(curr.name).date()
            current_price = float(curr["Close"])
            daily_volume = float(curr["Volume"])
            
            sent_today = sentiment_hist.loc[date_obj].to_dict() if not sentiment_hist.empty and date_obj in sentiment_hist.index else None
            risk_today = risk_hist.loc[date_obj].to_dict() if not risk_hist.empty and date_obj in risk_hist.index else None
            
            # Accrue interest
            if cfg.backtest.model_cash_interest:
                daily_rate = rf_annual / 252.0
                portfolio.cash += portfolio.cash * daily_rate
            
            if position == 0:
                # Look for entry using generate_signal
                sig_df = df.iloc[:i+1]
                sig = generate_signal(ticker, sig_df, cfg, sentiment_data=sent_today, risk_data=risk_today, precomputed_df=sig_df)
                
                if sig.action == Action.BUY:
                    # Buy!
                    kelly_val = float(risk_today.get("kelly_fraction_capped", 0.0)) if risk_today else None
                    shares_to_alloc = shares_to_buy(
                        sig=sig,
                        portfolio_nav=portfolio.equity({ticker: current_price}),
                        current_price=current_price,
                        cfg=cfg,
                        kelly_fraction=kelly_val,
                        daily_volume=daily_volume
                    )
                    if shares_to_alloc > 0:
                        exec_price = current_price * (1.0 + cfg.backtest.slippage)
                        if portfolio.buy(ticker, shares_to_alloc, exec_price, date_str):
                            position = 1
                            entry_price = exec_price
                            stop_loss = sig.stop_loss
                            bars_since_entry = 0
                            entry_atr = float(curr["atr"]) if "atr" in df.columns and pd.notna(curr.get("atr")) else 0.0
            elif position == 1:
                bars_since_entry += 1
                shares = portfolio.shares_held(ticker)

                # 0. False-breakout filter — early exit on reversal
                if (cfg.false_breakout.enabled and
                    bars_since_entry <= cfg.false_breakout.max_bars and
                    entry_atr > 0 and entry_price > 0):
                    reversal_threshold = entry_price - cfg.false_breakout.reversal_atr_mult * entry_atr
                    if current_price <= reversal_threshold:
                        exit_price = current_price * (1.0 - cfg.backtest.slippage)
                        portfolio.sell(ticker, shares, exit_price, date_str)
                        position = 0
                        portfolio.record_equity(date_str, {})
                        continue

                # 1. Stop loss hit
                if float(curr["Low"]) <= stop_loss:
                    exit_price = min(current_price, stop_loss) * (1.0 - cfg.backtest.slippage)
                    portfolio.sell(ticker, shares, exit_price, date_str)
                    position = 0
                # 2. Trailing EMA exit
                elif current_price < float(curr["ema_exit"]):
                    exit_price = current_price * (1.0 - cfg.backtest.slippage)
                    portfolio.sell(ticker, shares, exit_price, date_str)
                    position = 0
                    
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


def run_portfolio_backtest(
    cfg: VolatilityBreakoutConfig,
    ticker_ohlc: Dict[str, pd.DataFrame],
    benchmark_ohlc: Dict[str, pd.DataFrame],
    rf_annual: float = 0.04,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> BacktestSummary:
    """Run a full portfolio-level backtest aggregating across all tickers."""
    return run_backtest(cfg, ticker_ohlc, benchmark_ohlc, rf_annual, start_date, end_date)
