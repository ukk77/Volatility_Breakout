import sqlite3
import os
import logging
from contextlib import contextmanager
from typing import List, Dict

log = logging.getLogger(__name__)

DB_PATH = os.path.join(os.path.dirname(__file__), "vb_paper_trades.db")

@contextmanager
def _get_conn():
    conn = sqlite3.connect(DB_PATH, timeout=10.0)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.commit()
        conn.close()

def init_db():
    with _get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL UNIQUE,
                shares INTEGER NOT NULL,
                avg_cost REAL NOT NULL,
                stop_loss REAL NOT NULL,
                last_updated TEXT NOT NULL
            );
            
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                action TEXT NOT NULL,
                shares INTEGER NOT NULL,
                price REAL NOT NULL,
                reason TEXT,
                executed_at TEXT NOT NULL
            );
            
            CREATE TABLE IF NOT EXISTS run_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_date TEXT NOT NULL UNIQUE
            );
            
            CREATE TABLE IF NOT EXISTS portfolio_state (
                id INTEGER PRIMARY KEY,
                cash_balance REAL NOT NULL
            );
        """)
        # Initialize cash if empty
        cur = conn.execute("SELECT cash_balance FROM portfolio_state WHERE id=1")
        if not cur.fetchone():
            conn.execute("INSERT INTO portfolio_state (id, cash_balance) VALUES (1, 100000.0)")

def get_cash_balance() -> float:
    with _get_conn() as conn:
        cur = conn.execute("SELECT cash_balance FROM portfolio_state WHERE id=1")
        row = cur.fetchone()
        return float(row["cash_balance"]) if row else 0.0

def update_cash_balance(amount: float):
    with _get_conn() as conn:
        conn.execute("UPDATE portfolio_state SET cash_balance = ? WHERE id = 1", (amount,))

def has_run_today(date_str: str) -> bool:
    with _get_conn() as conn:
        cur = conn.execute("SELECT 1 FROM run_log WHERE run_date = ?", (date_str,))
        return bool(cur.fetchone())

def mark_run_today(date_str: str):
    with _get_conn() as conn:
        conn.execute("INSERT OR IGNORE INTO run_log (run_date) VALUES (?)", (date_str,))

def get_positions() -> List[Dict]:
    with _get_conn() as conn:
        cur = conn.execute("SELECT * FROM positions")
        return [dict(row) for row in cur.fetchall()]

def upsert_position(ticker: str, shares: int, avg_cost: float, stop_loss: float, date_str: str):
    with _get_conn() as conn:
        conn.execute("""
            INSERT INTO positions (ticker, shares, avg_cost, stop_loss, last_updated)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(ticker) DO UPDATE SET
                shares = excluded.shares,
                avg_cost = excluded.avg_cost,
                stop_loss = excluded.stop_loss,
                last_updated = excluded.last_updated
        """, (ticker, shares, avg_cost, stop_loss, date_str))

def remove_position(ticker: str):
    with _get_conn() as conn:
        conn.execute("DELETE FROM positions WHERE ticker = ?", (ticker,))

def log_trade(date_str: str, ticker: str, action: str, shares: int, price: float, reason: str):
    with _get_conn() as conn:
        conn.execute("""
            INSERT INTO trades (ticker, action, shares, price, reason, executed_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (ticker, action, shares, price, reason, date_str))

def get_trades(limit: int = 50) -> List[Dict]:
    with _get_conn() as conn:
        cur = conn.execute("SELECT * FROM trades ORDER BY executed_at DESC, id DESC LIMIT ?", (limit,))
        return [dict(row) for row in cur.fetchall()]
