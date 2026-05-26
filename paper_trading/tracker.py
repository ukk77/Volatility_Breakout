import logging
from datetime import datetime
from typing import Dict
from app.services.market_data import fetch_ohlcv
from . import db as paper_db
from ..config import VolatilityBreakoutConfig
from ..signals.generator import generate_signal, compute_indicators, Action
from ..position_sizing.sizer import shares_to_buy

log = logging.getLogger(__name__)

def run_paper_trading(cfg: VolatilityBreakoutConfig):
    paper_db.init_db()
    today_str = datetime.now().strftime("%Y-%m-%d")
    
    if paper_db.has_run_today(today_str):
        log.info(f"Paper trading already run for {today_str}. Skipping.")
        return
        
    positions = paper_db.get_positions()
    pos_map = {p["ticker"]: p for p in positions}
    
    log.info(f"Starting VB paper trading run for {today_str}. Checking {len(cfg.tickers)} tickers.")
    
    for ticker in cfg.tickers:
        try:
            ohlc = fetch_ohlcv(ticker, cfg.lookback_days)
            if len(ohlc) < 50:
                continue
                
            df = compute_indicators(ohlc, cfg)
            last_row = df.iloc[-1]
            current_price = float(last_row["Close"])
            ema_exit = float(last_row["ema_exit"])
            
            # 1. Manage Exits
            if ticker in pos_map:
                pos = pos_map[ticker]
                stop_loss = float(pos["stop_loss"])
                current_low = float(last_row["Low"])
                
                # Check Hard Stop
                if current_low <= stop_loss:
                    exit_price = min(current_price, stop_loss)
                    log.info(f"[{ticker}] HARD STOP HIT. Low {current_low:.2f} <= Stop {stop_loss:.2f}")
                    paper_db.log_trade(today_str, ticker, "SELL", pos["shares"], exit_price, "Hard Stop Hit")
                    paper_db.remove_position(ticker)
                    
                    cash = paper_db.get_cash_balance()
                    paper_db.update_cash_balance(cash + (pos["shares"] * exit_price))
                    continue
                    
                # Check Trailing EMA Stop
                if current_price < ema_exit:
                    log.info(f"[{ticker}] EMA EXIT HIT. Close {current_price:.2f} < 8-EMA {ema_exit:.2f}")
                    paper_db.log_trade(today_str, ticker, "SELL", pos["shares"], current_price, "Trailing EMA Exit")
                    paper_db.remove_position(ticker)
                    
                    cash = paper_db.get_cash_balance()
                    paper_db.update_cash_balance(cash + (pos["shares"] * current_price))
                    continue

            # 2. Look for Entries
            if ticker not in pos_map:
                sig = generate_signal(ticker, ohlc, cfg)
                if sig.action == Action.BUY:
                    cash = paper_db.get_cash_balance()
                    alloc = cash * (cfg.position_sizing.base_position_pct / 100.0)
                    shares = int(alloc / sig.price)
                    
                    if shares > 0:
                        cost = shares * sig.price
                        paper_db.update_cash_balance(cash - cost)
                        paper_db.upsert_position(ticker, shares, sig.price, sig.stop_loss, today_str)
                        paper_db.log_trade(today_str, ticker, "BUY", shares, sig.price, sig.reason)
                        log.info(f"[{ticker}] BREAKOUT! Bought {shares} shares @ {sig.price:.2f}. Stop: {sig.stop_loss:.2f}")
                        
        except Exception as e:
            log.error(f"[{ticker}] Paper trading error: {e}")
            
    paper_db.mark_run_today(today_str)
    log.info("VB Paper trading run complete.")
