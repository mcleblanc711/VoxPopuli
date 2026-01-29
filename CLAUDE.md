# VoxPopuli - Reddit Sentiment Trading Backtester

## Project Overview

VoxPopuli is a sentiment-based trading backtester that analyzes Reddit discussions to generate trading signals. It combines social sentiment analysis with historical price data to backtest various trading strategies.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLI (app.py)                            │
├─────────────────────────────────────────────────────────────────┤
│                      Strategy Engine                            │
│  ┌──────────────┬──────────────┬──────────────┬────────────────┐│
│  │  Attention   │  Sentiment   │  Contrarian  │ Cross-Subreddit││
│  │  Momentum    │  Divergence  │              │                ││
│  └──────────────┴──────────────┴──────────────┴────────────────┘│
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                   Velocity Sentiment                        ││
│  └─────────────────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────────────────┤
│                      Backtest Engine                            │
│  ┌──────────────┬──────────────┬──────────────────────────────┐│
│  │    Engine    │   Metrics    │          Export              ││
│  └──────────────┴──────────────┴──────────────────────────────┘│
├─────────────────────────────────────────────────────────────────┤
│                        Data Layer                               │
│  ┌──────────────┬──────────────┬──────────────┬────────────────┐│
│  │   Reddit     │    Price     │    Ticker    │   Sentiment    ││
│  │   Ingester   │   Fetcher    │   Extractor  │    Scorer      ││
│  └──────────────┴──────────────┴──────────────┴────────────────┘│
├─────────────────────────────────────────────────────────────────┤
│                    Data Storage (Parquet)                       │
│  ┌──────────────────────┬──────────────────────────────────────┐│
│  │    data/reddit/      │           data/prices/               ││
│  │   {sub}_{year}.parq  │         {ticker}_daily.parquet       ││
│  └──────────────────────┴──────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow

1. **Reddit Data Ingestion**
   - Download Arctic Shift Pushshift dumps (.zst compressed NDJSON)
   - Parse and extract: post_id, title, selftext, score, num_comments, created_utc
   - Store as parquet: `data/reddit/{subreddit}_{year}_{month}.parquet`

2. **Price Data Fetching**
   - Primary: IBKR TWS via ib_insync
   - Fallback: yfinance (crypto, backup)
   - Store as parquet: `data/prices/{ticker}_daily.parquet`

3. **Ticker Extraction**
   - Regex pattern: `$TICKER` format
   - Keyword matching against NYSE/NASDAQ list
   - Filter false positives: A, I, CEO, DD, etc.

4. **Sentiment Scoring**
   - VADER (default, fast)
   - FinBERT (optional, more accurate)
   - Score: title + selftext + top N comments
   - Output: compound sentiment (-1 to 1), comment velocity

5. **Strategy Execution**
   - Generate signals from sentiment data
   - Apply position sizing and risk management
   - Output trade list with entry/exit conditions

6. **Backtesting**
   - Vectorized execution for speed
   - Track positions, PnL, drawdowns
   - Calculate performance metrics

7. **Export & Visualization**
   - JSON export for dashboard
   - React dashboard with Recharts

## Module Reference

### Data Layer (`src/data/`)

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| `reddit_ingester.py` | Download/parse Reddit data | `download_dump()`, `parse_zst()`, `to_parquet()` |
| `price_fetcher.py` | Fetch OHLCV data | `fetch_ibkr()`, `fetch_yfinance()`, `cache_prices()` |
| `ticker_extractor.py` | Extract tickers from text | `extract_tickers()`, `filter_false_positives()` |
| `sentiment_scorer.py` | Score sentiment | `score_vader()`, `score_finbert()`, `score_post()` |

### Strategies (`src/strategies/`)

| Strategy | Logic |
|----------|-------|
| `attention_momentum.py` | Long top N tickers by mention count |
| `sentiment_divergence.py` | Long positive sentiment + flat price; short extended |
| `contrarian.py` | Short top sentiment, long bottom sentiment |
| `cross_subreddit.py` | Require N/5 subreddit consensus |
| `velocity_sentiment.py` | Entry on comment velocity spike + sentiment shift |

### Backtest (`src/backtest/`)

| Module | Purpose |
|--------|---------|
| `engine.py` | Core backtesting loop |
| `metrics.py` | Performance calculations |
| `export.py` | JSON export for dashboard |

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Fetch Reddit data
python src/app.py fetch-reddit --subreddits wallstreetbets,stocks --start 2020-01 --end 2023-12

# Fetch price data
python src/app.py fetch-prices --tickers AAPL,TSLA,GME --start 2020-01-01 --end 2023-12-31

# Run backtest
python src/app.py run-backtest --strategy attention_momentum --config config.yaml

# Export results
python src/app.py export-results --input data/results/backtest.pkl --output data/results/backtest.json

# Run tests
pytest tests/

# Dashboard development
cd dashboard && npm install && npm run dev

# Dashboard build
cd dashboard && npm run build
```

## Configuration (config.yaml)

```yaml
data:
  subreddits: [wallstreetbets, stocks, investing, options, stockmarket]
  date_range:
    start: "2020-01-01"
    end: "2023-12-31"

strategy:
  name: attention_momentum
  params:
    top_n: 10
    holding_period: 5
    min_mentions: 50

backtest:
  initial_capital: 100000
  position_size: 0.1
  max_positions: 10
  allow_shorts: false
  transaction_cost: 0.001

sentiment:
  model: vader  # or finbert
  include_comments: true
  top_n_comments: 10
```

## Key Data Schemas

### Reddit Post Parquet
```
post_id: string
title: string
selftext: string
score: int64
num_comments: int64
created_utc: timestamp
subreddit: string
```

### Price Parquet
```
date: date
open: float64
high: float64
low: float64
close: float64
volume: int64
adjusted_close: float64
source: string (ibkr|yfinance)
```

### Backtest Result JSON
```json
{
  "metadata": {
    "strategy": "attention_momentum",
    "date_range": ["2020-01-01", "2023-12-31"],
    "config": {}
  },
  "summary": {
    "total_return": 0.45,
    "cagr": 0.12,
    "sharpe_ratio": 1.5,
    "max_drawdown": -0.15,
    "win_rate": 0.55
  },
  "equity_curve": [...],
  "trades": [...],
  "monthly_returns": [...]
}
```

## Dependencies

- **Data**: pandas, pyarrow, zstandard, requests
- **Trading**: ib_insync, yfinance
- **Sentiment**: vaderSentiment, transformers (optional)
- **CLI**: typer, rich
- **Testing**: pytest
- **Dashboard**: React, Tailwind CSS, Recharts, Vite
