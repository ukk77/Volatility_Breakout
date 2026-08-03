# Archived one-off scripts

Moved out of the `volatility_breakout` root during workspace cleanup — not
referenced anywhere else in this repo:

- `run.py` — trivial `cli.main()` wrapper, inconsistent with `mean_reversion`/
  `trend_following`, which are invoked via `cli.py` directly. Use
  `python -m volatility_breakout.cli` (or `venv\Scripts\python.exe cli.py`)
  instead.
- `debug_signals.py` — ad-hoc indicator inspection script for NVDA. Still
  runnable standalone from this directory if needed.
