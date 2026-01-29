# VoxPopuli - Reddit Sentiment Trading Backtester

## Project Overview

VoxPopuli is a sentiment-based trading backtester that analyzes Reddit discussions to generate trading signals. It combines social sentiment analysis with historical price data to backtest various trading strategies.

**Live Dashboard**: https://mcleblanc711.github.io/VoxPopuli/

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Create test data
python scripts/create_test_data.py

# 3. Fetch price data
python src/app.py fetch-prices --tickers AAPL,TSLA,GME,AMD,NVDA,SPY --start 2023-01-01 --end 2023-12-31

# 4. Run a backtest
python src/app.py run-backtest -s attention_momentum --start 2023-01-01 --end 2023-12-31

# 5. View dashboard locally
cd dashboard && npm install && npm run dev
```

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

## CLI Commands

```bash
# List all available strategies
python src/app.py list-strategies

# Fetch Reddit data (downloads Arctic Shift dumps - large files)
python src/app.py fetch-reddit --subreddits wallstreetbets,stocks --start 2023-01 --end 2023-12

# Fetch price data from yfinance
python src/app.py fetch-prices --tickers AAPL,TSLA,GME --start 2023-01-01 --end 2023-12-31

# Run backtest with specific strategy
python src/app.py run-backtest -s attention_momentum --start 2023-01-01 --end 2023-12-31
python src/app.py run-backtest -s sentiment_divergence --start 2023-01-01 --end 2023-12-31
python src/app.py run-backtest -s contrarian --start 2023-01-01 --end 2023-12-31
python src/app.py run-backtest -s velocity_sentiment --start 2023-01-01 --end 2023-12-31

# Analyze sentiment in Reddit data
python src/app.py analyze-sentiment --input data/reddit/test.parquet --model vader

# Extract and count ticker mentions
python src/app.py extract-tickers --input data/reddit/test.parquet --top 20

# Run tests
pytest tests/
```

## Data Flow

1. **Reddit Data Ingestion** (`src/data/reddit_ingester.py`)
   - Download Arctic Shift Pushshift dumps (.zst compressed NDJSON)
   - Parse and extract: post_id, title, selftext, score, num_comments, created_utc
   - Store as parquet: `data/reddit/{subreddit}_{year}_{month}.parquet`

2. **Price Data Fetching** (`src/data/price_fetcher.py`)
   - Primary: IBKR TWS via ib_insync
   - Fallback: yfinance (default, works without setup)
   - Store as parquet: `data/prices/{ticker}_daily.parquet`

3. **Ticker Extraction** (`src/data/ticker_extractor.py`)
   - Regex pattern: `$TICKER` format
   - Keyword matching against NYSE/NASDAQ list
   - Filter false positives: CEO, DD, YOLO, etc.

4. **Sentiment Scoring** (`src/data/sentiment_scorer.py`)
   - VADER (default, fast)
   - FinBERT (optional, requires transformers)
   - Score: title + selftext + top N comments
   - Output: compound sentiment (-1 to 1)

5. **Strategy Execution** (`src/strategies/`)
   - Generate signals from sentiment data
   - Apply position sizing and risk management
   - Output trade list with entry/exit conditions

6. **Backtesting** (`src/backtest/engine.py`)
   - Vectorized execution for speed
   - Track positions, PnL, drawdowns
   - Calculate performance metrics

7. **Export & Visualization** (`src/backtest/export.py`)
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

| Strategy | Logic | Key Parameters |
|----------|-------|----------------|
| `attention_momentum.py` | Long top N tickers by mention count | `top_n`, `min_mentions`, `holding_period` |
| `sentiment_divergence.py` | Long positive sentiment + flat price; short extended | `sentiment_threshold`, `price_change_threshold` |
| `contrarian.py` | Short top sentiment, long bottom sentiment | `extreme_threshold`, `min_mentions` |
| `cross_subreddit.py` | Require N subreddit consensus | `min_subreddits`, `min_mentions_per_sub` |
| `velocity_sentiment.py` | Entry on comment velocity spike + sentiment shift | `velocity_threshold`, `sentiment_shift_threshold` |

### Backtest (`src/backtest/`)

| Module | Purpose |
|--------|---------|
| `engine.py` | Core backtesting loop, position management |
| `metrics.py` | Performance calculations (Sharpe, Sortino, drawdown) |
| `export.py` | JSON export for dashboard |

## Configuration (config.yaml)

```yaml
data:
  subreddits:
    - wallstreetbets
    - stocks
    - investing
  date_range:
    start: "2023-01-01"
    end: "2023-12-31"

strategy:
  name: attention_momentum
  params:
    top_n: 10
    holding_period: 5
    min_mentions: 5  # Lower for testing

backtest:
  initial_capital: 100000
  position_size: 0.1
  max_positions: 10
  allow_shorts: false
  transaction_cost: 0.001

sentiment:
  model: vader  # or finbert
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
    "date_range": ["2023-01-01", "2023-12-31"],
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
  "monthly_returns": [...],
  "monthly_heatmap": {...}
}
```

## Dashboard

The React dashboard (`dashboard/`) provides:
- **Summary Dashboard**: Key metrics cards, equity curve chart, monthly returns heatmap
- **Trade Explorer**: Filterable/sortable table of all trades
- **Strategy Comparison**: Overlay multiple strategy results
- **Configuration Display**: View backtest parameters

### Local Development
```bash
cd dashboard
npm install
npm run dev
# Open http://localhost:5173
```

### Deployment
The dashboard is automatically deployed to GitHub Pages via `.github/workflows/deploy.yml` on push to master.

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src

# Run specific test file
pytest tests/test_strategies.py -v
```

## Project Files

```
VoxPopuli/
├── .github/workflows/
│   └── deploy.yml          # GitHub Pages deployment
├── src/
│   ├── __init__.py
│   ├── app.py              # CLI application (typer)
│   ├── data/
│   │   ├── reddit_ingester.py
│   │   ├── price_fetcher.py
│   │   ├── ticker_extractor.py
│   │   └── sentiment_scorer.py
│   ├── strategies/
│   │   ├── base_strategy.py
│   │   ├── attention_momentum.py
│   │   ├── sentiment_divergence.py
│   │   ├── contrarian.py
│   │   ├── cross_subreddit.py
│   │   └── velocity_sentiment.py
│   └── backtest/
│       ├── engine.py
│       ├── metrics.py
│       └── export.py
├── scripts/
│   └── create_test_data.py # Generate synthetic Reddit data
├── dashboard/
│   ├── src/
│   │   ├── App.tsx
│   │   ├── components/
│   │   │   ├── SummaryDashboard.tsx
│   │   │   ├── TradeExplorer.tsx
│   │   │   ├── StrategyComparison.tsx
│   │   │   └── ConfigDisplay.tsx
│   │   └── types.ts
│   └── public/data/        # Backtest results for dashboard
├── tests/
│   ├── test_sentiment.py
│   ├── test_strategies.py
│   ├── test_backtest.py
│   └── test_ticker_extractor.py
├── data/
│   ├── reddit/             # Reddit parquet files
│   ├── prices/             # Price parquet files
│   └── results/            # Backtest JSON results
├── config.yaml             # Default configuration
├── requirements.txt        # Python dependencies
├── README.md               # User documentation
├── CLAUDE.md               # This file - development docs
└── TODO.md                 # Task tracker
```

## Dependencies

### Python
- **Data**: pandas, pyarrow, zstandard, requests
- **Trading**: ib_insync, yfinance
- **Sentiment**: vaderSentiment, transformers (optional)
- **CLI**: typer, rich
- **Testing**: pytest

### Dashboard
- React 18, Vite, TypeScript
- Tailwind CSS
- Recharts

## Common Issues

### "Module not found" when running CLI
Run from the project root directory, not from `src/` or `dashboard/`.

### No trades generated
- Check `min_mentions` in config.yaml (lower it for testing)
- Ensure price data exists for the tickers mentioned in Reddit data
- Check date ranges align between Reddit data and price data

### Pandas frequency deprecation warnings
Use lowercase frequency aliases: `'h'` instead of `'H'`, `'d'` instead of `'D'`

### Dashboard 404 on GitHub Pages
1. Go to repo Settings > Pages
2. Set Source to "GitHub Actions"
3. Wait a few minutes for deployment
