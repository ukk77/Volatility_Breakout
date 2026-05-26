import argparse
import logging
import json
import sys
import os

# Add risk_calculator/backend to path so we can import app.services.market_data
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "risk_calculator", "backend"))

from .config import VolatilityBreakoutConfig
from .backtest.engine import run_backtest

log = logging.getLogger(__name__)


def cmd_backtest(args) -> None:
    from app.services.market_data import fetch_ohlcv
    
    cfg = VolatilityBreakoutConfig()
    tickers = [args.ticker.upper()] if args.ticker else cfg.tickers
    
    print("Loading price data...")
    ticker_ohlc = {}
    for ticker in tickers:
        try:
            ticker_ohlc[ticker] = fetch_ohlcv(ticker, cfg.lookback_days)
            print(f"  {ticker}: {len(ticker_ohlc[ticker])} rows")
        except Exception as e:
            print(f"  {ticker}: FAILED - {e}")
            
    benchmark_names = [cfg.backtest.benchmark_ticker] + list(ticker_ohlc.keys())
    benchmark_ohlc = {}
    for b in benchmark_names:
        if b not in benchmark_ohlc:
            try:
                benchmark_ohlc[b] = fetch_ohlcv(b, cfg.lookback_days)
            except:
                pass
                
    rf = 0.04
    
    print(f"\nRunning Volatility Breakout Backtest | capital=${cfg.backtest.initial_capital:,.0f}\n")
    
    summary = run_backtest(
        cfg=cfg,
        ticker_ohlc=ticker_ohlc,
        benchmark_ohlc=benchmark_ohlc,
        rf_annual=rf,
        start_date=args.start,
        end_date=args.end,
    )
    
    def _fmt(v, decimals=2):
        if v is None or abs(v) > 9999:
            return "N/A"
        return f"{v:.{decimals}f}"

    bench = cfg.backtest.benchmark_ticker.lower()
    print(
        f"{'TICKER':<10} {'RETURN%':>8} {'CAGR%':>7} {'SHARPE':>7} {'CALMAR':>7} "
        f"{'MAX_DD%':>8} {'PF':>6} {'AVG_HOLD':>9} {'TRADES':>7} {'WIN%':>6} {'ALPHA_C+3%':>11}"
    )
    print("-" * 92)

    rows = list(summary.results.items())
    if summary.portfolio_metrics:
        rows.append(("PORTFOLIO", type("R", (), {"metrics": summary.portfolio_metrics})()))

    for ticker, result in rows:
        m = result.metrics
        hold_str = f"{m.get('avg_holding_days') or 0:.0f}d"
        alpha_c3 = m.get("alpha_vs_cash_plus_3_pct")
        print(
            f"{ticker:<10} {m['total_return_pct']:>8.1f} {m['cagr_pct']:>7.1f} "
            f"{_fmt(m.get('sharpe')):>7} {_fmt(m.get('calmar')):>7} "
            f"{m['max_drawdown_pct']:>8.1f} {_fmt(m.get('profit_factor')):>6} "
            f"{hold_str:>9} "
            f"{m['total_trades']:>7} {m['win_rate_pct']:>6.1f} {_fmt(alpha_c3):>11}"
        )


def cmd_paper(args) -> None:
    """Run paper trading — process today's signals and update positions."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    from volatility_breakout.paper_trading.tracker import run_paper_trading
    
    cfg = VolatilityBreakoutConfig()
    run_paper_trading(cfg)


def cmd_positions(args) -> None:
    """Show current open paper positions and recent trades."""
    from app.services.market_data import fetch_ohlcv
    from volatility_breakout.paper_trading import db as paper_db
    
    positions = paper_db.get_positions()
    cash = paper_db.get_cash_balance()
    
    print(f"\nVolatility Breakout Paper Trading")
    print(f"Cash Balance: ${cash:,.2f}")
    print(f"Open Positions: {len(positions)}")
    print("-" * 65)
    
    if not positions:
        print("  (No open positions)")
    else:
        print(f"{'TICKER':<8} {'SHARES':>6} {'AVG_COST':>10} {'CUR_PRICE':>10} {'UNREAL_PNL':>12}")
        print("-" * 65)
        for p in positions:
            try:
                ohlc = fetch_ohlcv(p["ticker"], 5)
                cur_price = float(ohlc["Close"].iloc[-1])
            except Exception:
                cur_price = p["avg_cost"]
                
            unreal_pnl = (cur_price - p["avg_cost"]) * p["shares"]
            print(
                f"{p['ticker']:<8} {p['shares']:>6} {p['avg_cost']:>10.2f} "
                f"{cur_price:>10.2f} {unreal_pnl:>+12.2f}"
            )
            
    print("\nRecent Trades:")
    trades = paper_db.get_trades(limit=10)
    if not trades:
        print("  (No recent trades)")
    else:
        for t in trades:
            print(f"  [{t['action']}] {t['ticker']:<5} {t['shares']} shares @ ${t['price']:.2f} ({t['executed_at']})")
    print()


def cmd_signals(args) -> None:
    """Print current BUY/SELL/HOLD signals."""
    from app.services.market_data import fetch_ohlcv
    from volatility_breakout.signals.generator import generate_signal

    cfg = VolatilityBreakoutConfig()
    tickers = [args.ticker.upper()] if args.ticker else cfg.tickers

    print(f"\nScanning Volatility Breakout Signals")
    print("-" * 100)
    print(f"{'TICKER':<8} {'ACTION':<6} {'PRICE':>9} {'STOP':>9} {'REASON'}")
    print("-" * 100)

    for ticker in tickers:
        try:
            ohlc = fetch_ohlcv(ticker, cfg.lookback_days)
            sig = generate_signal(ticker, ohlc, cfg)
            
            color_code = ""
            if sig.action == "BUY":
                color_code = "\033[92m"  # Green
            elif sig.action == "SELL":
                color_code = "\033[91m"  # Red
            
            end_color = "\033[0m" if color_code else ""
            
            print(f"{color_code}{sig.ticker:<8} {sig.action.name:<6} "
                  f"{sig.price:>9.2f} {sig.stop_loss:>9.2f} {sig.reason}{end_color}")
        except Exception as e:
            print(f"{ticker:<8} ERROR  {str(e)}")
    print()

def main():
    parser = argparse.ArgumentParser(description="Volatility Breakout strategy")
    subs = parser.add_subparsers(dest="command", required=True)
    
    p_sig = subs.add_parser("signals", help="Show current signals")
    p_sig.add_argument("--ticker", help="Single ticker")
    p_sig.set_defaults(func=cmd_signals)

    p_bt = subs.add_parser("backtest", help="Run backtest")
    p_bt.add_argument("--ticker", help="Single ticker")
    p_bt.add_argument("--start", help="Start date")
    p_bt.add_argument("--end", help="End date")
    p_bt.set_defaults(func=cmd_backtest)
    
    p_paper = subs.add_parser("paper", help="Run daily paper trading")
    p_paper.set_defaults(func=cmd_paper)
    
    p_pos = subs.add_parser("positions", help="Show open paper positions")
    p_pos.set_defaults(func=cmd_positions)
    
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
