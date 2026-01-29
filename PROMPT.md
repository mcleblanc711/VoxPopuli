# VoxPopuli: Reddit Sentiment-Based Trading Backtester

## Project Overview
Build a local Python application + GitHub Pages dashboard for backtesting trading strategies based on Reddit sentiment analysis. The system ingests historical Reddit data (Pushshift/Arctic Shift), scores sentiment from posts and comments, maps discussions to tickers, and backtests configurable strategies against historical price data.

## Architecture

### Local Python App
- Data ingestion pipeline (Pushshift dumps + live Reddit API for forward collection)
- Price data cache (IBKR TWS via ib_insync for equities, yfinance fallback for crypto)
- Sentiment scoring engine (using Reddit .json comment structure)
- Strategy backtester with selectable strategies
- JSON export for dashboard consumption

### GitHub Pages Dashboard (separate repo or /docs folder)
- Interactive visualization of backtest results
- Strategy comparison views
- Equity curves, drawdown charts, per-trade breakdown
- Subreddit/ticker/date range filters

## Documentation Requirements

### CLAUDE.md
Comprehensive project documentation for AI-assisted development. Include:
- Project purpose and architecture overview
- How each module connects (data flow diagram in ASCII or mermaid)
- Key design decisions and rationale
- File-by-file breakdown of responsibilities
- Common commands (run backtest, fetch data, export results, deploy dashboard)
- Environment setup instructions (IBKR TWS config, API keys, dependencies)
- Testing approach and how to run tests
- Known limitations and assumptions

### TODO.md
Living task tracker organized by priority:
- **Immediate:** Current sprint / blocking items
- **Short-term:** Next features to implement
- **Long-term:** Nice-to-haves, optimizations, stretch goals
- **Technical debt:** Refactors, cleanup, performance improvements
- Include checkboxes for tracking completion
- Note dependencies between tasks where relevant

### SCRATCHPAD.md
Informal planning document for future development:
- Ideas under consideration (not committed to TODO yet)
- Architecture questions to revisit
- Alternative approaches explored and why they were/weren't chosen
- Performance observations and optimization ideas
- Notes from backtesting experiments (what worked, what didn't)
- Links to relevant papers, repos, or resources for future reference

## Data Pipeline

### Reddit Data
**Subreddits:** r/wallstreetbets, r/stocks, r/cryptocurrency, r/baystreetbets, r/investing

**Source:** Arctic Shift Pushshift dumps (https://arctic-shift.photon-reddit.com/)
- Coverage: 2015 – June 2023
- Store as parquet: `/{subreddit}_{year}_{month}.parquet`
- Fields needed: post_id, title, selftext, score, num_comments, created_utc, comments (nested with body, score, created_utc)

**Sentiment Scoring:**
- Extract from post title + selftext + top N comments
- Use VADER or FinBERT (make selectable)
- Output: compound sentiment score (-1 to 1), comment velocity (comments/hour in first 24h)

**Ticker Extraction:**
- Regex for $TICKER format
- Keyword matching against known ticker list (fetch from IBKR or use static NYSE/NASDAQ list)
- Filter out false positives (common words: A, I, CEO, DD, etc.)

### Price Data
**Primary:** IBKR TWS via ib_insync
- Daily OHLCV bars
- Cache locally as parquet: `/{ticker}_daily.parquet`
- Include dividends/splits adjusted close

**Fallback (crypto + backup):** yfinance
- BTC-USD, ETH-USD, and any tickers IBKR lacks

**Schema:**
```
date, open, high, low, close, volume, adjusted_close, source
```

## Strategy Engine

Implement all 5 as selectable options with configurable parameters:

### 1. Attention Momentum
- Long top N tickers by mention count
- Rebalance weekly (configurable: daily/weekly/monthly)
- Parameters: N (portfolio size), holding period, minimum mention threshold

### 2. Sentiment Divergence
- Long: positive sentiment + price hasn't moved (< X% in prior week)
- Short: extreme positive sentiment + price already extended (> Y% run)
- Parameters: sentiment threshold, price move thresholds, lookback window

### 3. Contrarian / Mean Reversion
- Short top sentiment, long bottom sentiment
- Parameters: sentiment percentile thresholds, holding period

### 4. Cross-Subreddit Consensus
- Require N of 5 subreddits to agree on direction before entry
- Parameters: consensus threshold (e.g., 3/5), sentiment alignment definition

### 5. Comment Velocity + Sentiment
- Entry trigger: spike in comment velocity (> X std dev) + sentiment shift
- Parameters: velocity threshold, sentiment delta threshold, lookback for baseline

### Common Parameters (all strategies):
- Date range (start/end)
- Subreddit selection (multi-select)
- Position sizing (equal weight vs. sentiment-weighted)
- Max positions
- Allow shorts (boolean)
- Transaction costs (bps)

## Backtest Engine

- Event-driven or vectorized (vectorized preferred for speed)
- Track: entry/exit dates, entry/exit prices, position size, PnL, sentiment score at entry
- Calculate: total return, CAGR, Sharpe ratio, Sortino ratio, max drawdown, win rate, profit factor
- Benchmark comparison (SPY buy-and-hold)
- Export results as JSON for dashboard

## Output Schema (for dashboard)
```json
{
  "metadata": {
    "strategy": "attention_momentum",
    "params": {...},
    "date_range": ["2018-01-01", "2023-06-01"],
    "subreddits": ["wallstreetbets", "stocks"],
    "generated_at": "2025-01-28T12:00:00Z"
  },
  "summary": {
    "total_return": 0.45,
    "cagr": 0.12,
    "sharpe": 1.4,
    "sortino": 1.8,
    "max_drawdown": -0.18,
    "win_rate": 0.58,
    "total_trades": 234,
    "benchmark_return": 0.32
  },
  "equity_curve": [
    {"date": "2018-01-01", "equity": 100000, "benchmark": 100000},
    ...
  ],
  "trades": [
    {
      "ticker": "GME",
      "direction": "long",
      "entry_date": "2021-01-25",
      "exit_date": "2021-02-01",
      "entry_price": 76.79,
      "exit_price": 325.00,
      "pnl": 0.323,
      "sentiment_at_entry": 0.82,
      "subreddit_source": ["wallstreetbets"]
    },
    ...
  ],
  "monthly_returns": [
    {"month": "2018-01", "return": 0.02, "benchmark": 0.01},
    ...
  ]
}
```

## Dashboard Requirements

**Tech:** React (or vanilla JS if simpler), Tailwind CSS, Recharts or Plotly for charts

**Views:**

1. **Summary Dashboard**
   - Key metrics cards (return, Sharpe, drawdown, etc.)
   - Equity curve (strategy vs. benchmark)
   - Monthly returns heatmap

2. **Trade Explorer**
   - Filterable/sortable table of all trades
   - Click to expand: show sentiment context, Reddit post links if available

3. **Strategy Comparison**
   - Upload/select multiple result JSONs
   - Side-by-side metrics table
   - Overlaid equity curves

4. **Configuration Display**
   - Show parameters used for the backtest
   - Subreddit weights, date range, strategy settings

**Interactivity:**
- Date range slider to zoom equity curve
- Ticker filter dropdown
- Subreddit filter checkboxes

## File Structure
```
voxpopuli/
├── CLAUDE.md                   # Comprehensive project docs for AI-assisted dev
├── TODO.md                     # Prioritized task tracker with checkboxes
├── SCRATCHPAD.md               # Informal planning and future ideas
├── README.md                   # User-facing project overview
├── requirements.txt
├── config.yaml                 # Default params, API keys placeholder
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── reddit_ingester.py  # Pushshift download + parse
│   │   ├── price_fetcher.py    # IBKR + yfinance
│   │   ├── ticker_extractor.py # Regex + filtering
│   │   └── sentiment_scorer.py # VADER / FinBERT
│   ├── strategies/
│   │   ├── base_strategy.py
│   │   ├── attention_momentum.py
│   │   ├── sentiment_divergence.py
│   │   ├── contrarian.py
│   │   ├── cross_subreddit.py
│   │   └── velocity_sentiment.py
│   ├── backtest/
│   │   ├── engine.py
│   │   ├── metrics.py
│   │   └── export.py
│   └── app.py                  # CLI or simple GUI to run backtests
├── data/
│   ├── reddit/                 # Parquet files
│   ├── prices/                 # Parquet files
│   └── results/                # Exported JSONs
├── dashboard/                  # GitHub Pages deployable
│   ├── index.html
│   ├── src/
│   │   ├── App.jsx
│   │   ├── components/
│   │   └── utils/
│   ├── public/
│   │   └── sample_results.json
│   └── package.json
└── tests/
    ├── test_sentiment.py
    ├── test_strategies.py
    └── test_backtest.py
```

## Tech Stack

**Python:**
- ib_insync (IBKR connection)
- yfinance (fallback price data)
- pandas, numpy (data manipulation)
- pyarrow (parquet I/O)
- vaderSentiment or transformers (FinBERT) for sentiment
- pyyaml (config)
- typer or click (CLI)
- pytest (testing)

**Dashboard:**
- React 18
- Tailwind CSS
- Recharts or Plotly.js
- Vite (build tool)

## Implementation Order

1. **Data layer first:** Reddit ingester + price fetcher with caching
2. **Sentiment + ticker extraction:** Get this validated before strategies
3. **Single strategy (Attention Momentum):** End-to-end backtest working
4. **Remaining strategies:** Add incrementally
5. **Dashboard MVP:** Equity curve + summary metrics
6. **Dashboard polish:** Trade explorer, comparison view
7. **Testing + documentation**

## Notes

- Assume user has IBKR TWS running locally with API enabled (port 7497 paper, 7496 live)
- Reddit API credentials optional (only needed for forward collection, not historical)
- FinBERT is heavier but more accurate for financial text — make it optional with VADER as default
- Include sample result JSONs in dashboard for demo purposes