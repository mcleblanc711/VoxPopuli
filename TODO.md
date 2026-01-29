# VoxPopuli TODO

## Phase 1: Project Scaffolding
- [x] Create directory structure
- [x] Create CLAUDE.md
- [x] Create TODO.md
- [x] Create SCRATCHPAD.md
- [x] Create README.md
- [x] Create requirements.txt
- [x] Create config.yaml

## Phase 2: Data Layer
- [x] Implement reddit_ingester.py
  - [x] Download Arctic Shift dumps
  - [x] Parse zst-compressed NDJSON
  - [x] Convert to parquet format
- [x] Implement price_fetcher.py
  - [x] IBKR TWS connection
  - [x] yfinance fallback
  - [x] Parquet caching
- [x] Implement ticker_extractor.py
  - [x] $TICKER regex pattern
  - [x] NYSE/NASDAQ keyword matching
  - [x] False positive filtering
- [x] Implement sentiment_scorer.py
  - [x] VADER scorer
  - [x] FinBERT scorer (optional)
  - [x] Post scoring pipeline

## Phase 3: Strategy Engine
- [x] Implement base_strategy.py
- [x] Implement attention_momentum.py
- [x] Implement sentiment_divergence.py
- [x] Implement contrarian.py
- [x] Implement cross_subreddit.py
- [x] Implement velocity_sentiment.py

## Phase 4: Backtest Engine
- [x] Implement engine.py
- [x] Implement metrics.py
- [x] Implement export.py

## Phase 5: CLI Application
- [x] Implement app.py with typer
  - [x] fetch-reddit command
  - [x] fetch-prices command
  - [x] run-backtest command
  - [x] export-results command

## Phase 6: Dashboard
- [x] Set up Vite + React + Tailwind
- [x] Create Summary Dashboard
- [x] Create Trade Explorer
- [x] Create Strategy Comparison
- [x] Create Configuration Display
- [ ] GitHub Pages deployment setup

## Phase 7: Testing
- [x] test_sentiment.py
- [x] test_strategies.py
- [x] test_backtest.py

## Verification
- [x] Verify module imports
- [ ] Verify CLI works (requires dependencies)
- [ ] Verify tests pass (requires dependencies)
- [ ] Verify dashboard builds (requires npm install)
