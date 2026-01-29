# VoxPopuli

A Reddit sentiment-based trading backtester with a Python backend and React dashboard.

## Overview

VoxPopuli analyzes Reddit discussions to generate trading signals based on social sentiment. It supports multiple trading strategies and provides a comprehensive backtesting framework with visualization.

## Features

- **Reddit Data Ingestion**: Download and parse Arctic Shift Pushshift dumps
- **Price Data**: IBKR TWS integration with yfinance fallback
- **Sentiment Analysis**: VADER (fast) and FinBERT (accurate) options
- **5 Trading Strategies**:
  - Attention Momentum: Long most-mentioned tickers
  - Sentiment Divergence: Trade sentiment/price divergences
  - Contrarian: Fade extreme sentiment
  - Cross-Subreddit: Require multi-subreddit consensus
  - Velocity Sentiment: Trade on comment velocity spikes
- **Comprehensive Metrics**: Sharpe, Sortino, max drawdown, win rate, etc.
- **React Dashboard**: Interactive visualization with Recharts

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/voxpopuli.git
cd voxpopuli

# Install Python dependencies
pip install -r requirements.txt

# Install dashboard dependencies
cd dashboard
npm install
```

## Quick Start

```bash
# 1. Fetch Reddit data
python src/app.py fetch-reddit --subreddits wallstreetbets,stocks --start 2022-01 --end 2023-12

# 2. Fetch price data for extracted tickers
python src/app.py fetch-prices --tickers AAPL,TSLA,GME --start 2022-01-01 --end 2023-12-31

# 3. Run a backtest
python src/app.py run-backtest --strategy attention_momentum --config config.yaml

# 4. View results in dashboard
cd dashboard && npm run dev
```

## Configuration

Edit `config.yaml` to customize:

```yaml
data:
  subreddits: [wallstreetbets, stocks, investing]
  date_range:
    start: "2022-01-01"
    end: "2023-12-31"

strategy:
  name: attention_momentum
  params:
    top_n: 10
    holding_period: 5

backtest:
  initial_capital: 100000
  position_size: 0.1
  max_positions: 10
```

## Project Structure

```
voxpopuli/
├── src/
│   ├── data/           # Data ingestion modules
│   ├── strategies/     # Trading strategy implementations
│   ├── backtest/       # Backtesting engine
│   └── app.py          # CLI application
├── data/
│   ├── reddit/         # Reddit data (parquet)
│   ├── prices/         # Price data (parquet)
│   └── results/        # Backtest results
├── dashboard/          # React dashboard
└── tests/              # Test suite
```

## Strategies

| Strategy | Description | Best For |
|----------|-------------|----------|
| Attention Momentum | Long top N by mentions | Trending stocks |
| Sentiment Divergence | Trade sentiment/price gaps | Mean reversion |
| Contrarian | Fade extreme sentiment | Overbought/oversold |
| Cross-Subreddit | Multi-sub consensus | High conviction |
| Velocity Sentiment | Comment velocity spikes | Breakouts |

## Dashboard

The React dashboard provides:
- Performance summary with key metrics
- Interactive equity curve
- Monthly returns heatmap
- Filterable trade explorer
- Strategy comparison overlay

## Testing

```bash
pytest tests/
```

## License

MIT License
