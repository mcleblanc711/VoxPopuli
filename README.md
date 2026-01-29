# VoxPopuli

A Reddit sentiment-based trading backtester with a Python backend and React dashboard.

**[View Live Dashboard](https://mcleblanc711.github.io/VoxPopuli/)** | [Documentation](CLAUDE.md)

## Overview

VoxPopuli analyzes Reddit discussions to generate trading signals based on social sentiment. It supports multiple trading strategies and provides a comprehensive backtesting framework with visualization.

The project explores the relationship between retail investor sentiment on social media and stock price movements—a phenomenon that gained widespread attention during the 2021 GameStop short squeeze.

## Features

- **Reddit Data Ingestion**: Download and parse Arctic Shift Pushshift dumps
- **Price Data**: IBKR TWS integration with yfinance fallback
- **Sentiment Analysis**: VADER (fast) and FinBERT (accurate) options
- **5 Trading Strategies**: Based on academic research and market microstructure theory
- **Comprehensive Metrics**: Sharpe, Sortino, max drawdown, win rate, etc.
- **React Dashboard**: Interactive visualization with Recharts

## Installation

```bash
# Clone the repository
git clone https://github.com/mcleblanc711/VoxPopuli.git
cd VoxPopuli

# Install Python dependencies
pip install -r requirements.txt

# Install dashboard dependencies
cd dashboard
npm install
```

## Quick Start

```bash
# 1. Create synthetic test data (or fetch real Reddit data)
python scripts/create_test_data.py

# 2. Fetch price data
python src/app.py fetch-prices --tickers AAPL,TSLA,GME,AMD,NVDA,SPY --start 2023-01-01 --end 2023-12-31

# 3. Run a backtest
python src/app.py run-backtest --strategy attention_momentum --start 2023-01-01 --end 2023-12-31

# 4. View results in dashboard
cd dashboard && npm run dev
```

## Trading Strategies

### 1. Attention Momentum

**Logic**: Long the top N most-mentioned tickers, assuming retail attention drives short-term price momentum.

**Research Basis**:
- Barber & Odean (2008) found that individual investors are net buyers of "attention-grabbing" stocks
- Da, Engelberg & Gao (2011) showed that increased Google search volume predicts higher stock prices in the following weeks
- Social media mentions serve as a proxy for retail investor attention

**Parameters**: `top_n`, `min_mentions`, `holding_period`

### 2. Sentiment Divergence

**Logic**: Trade when sentiment and price diverge—long when sentiment is positive but price is flat/down (undervalued), short when sentiment is negative but price is up (overvalued).

**Research Basis**:
- Baker & Wurgler (2006) demonstrated that investor sentiment predicts cross-sectional stock returns
- Divergences between sentiment and price may indicate mispricing that will correct
- Similar to classic technical analysis divergence patterns (RSI, MACD)

**Parameters**: `sentiment_threshold`, `price_change_threshold`, `lookback_days`

### 3. Contrarian

**Logic**: Fade extreme sentiment—short when the crowd is euphoric, long when fearful. Based on the principle that extreme sentiment often precedes reversals.

**Research Basis**:
- De Long et al. (1990) "Noise Trader Risk" paper showed that sentiment-driven traders can push prices away from fundamentals
- Tetlock (2007) found that high media pessimism predicts downward pressure on prices followed by reversals
- Warren Buffett's maxim: "Be fearful when others are greedy, and greedy when others are fearful"

**Parameters**: `extreme_threshold`, `min_mentions`

### 4. Cross-Subreddit Consensus

**Logic**: Only trade when multiple independent communities (subreddits) agree on a ticker, filtering out noise and echo chamber effects.

**Research Basis**:
- Wisdom of crowds theory (Surowiecki, 2004) suggests aggregating diverse opinions improves accuracy
- Chen et al. (2014) found that the aggregated opinion from Seeking Alpha predicts stock returns
- Cross-validation across communities reduces the impact of coordinated manipulation

**Parameters**: `min_subreddits`, `min_mentions_per_sub`, `sentiment_alignment_threshold`

### 5. Velocity Sentiment

**Logic**: Enter on sudden spikes in comment activity combined with sentiment shifts—capturing breakout moments when a stock suddenly captures retail attention.

**Research Basis**:
- Antweiler & Frank (2004) found that message volume predicts volatility
- Rapid increases in social media activity often precede significant price moves
- Combines volume analysis principles with sentiment confirmation

**Parameters**: `velocity_threshold`, `sentiment_shift_threshold`, `lookback_days`

## References

- Antweiler, W., & Frank, M. Z. (2004). Is all that talk just noise? The information content of internet stock message boards. *The Journal of Finance*, 59(3), 1259-1294.
- Baker, M., & Wurgler, J. (2006). Investor sentiment and the cross-section of stock returns. *The Journal of Finance*, 61(4), 1645-1680.
- Barber, B. M., & Odean, T. (2008). All that glitters: The effect of attention and news on the buying behavior of individual and institutional investors. *The Review of Financial Studies*, 21(2), 785-818.
- Chen, H., De, P., Hu, Y. J., & Hwang, B. H. (2014). Wisdom of crowds: The value of stock opinions transmitted through social media. *The Review of Financial Studies*, 27(5), 1367-1403.
- Da, Z., Engelberg, J., & Gao, P. (2011). In search of attention. *The Journal of Finance*, 66(5), 1461-1499.
- De Long, J. B., Shleifer, A., Summers, L. H., & Waldmann, R. J. (1990). Noise trader risk in financial markets. *Journal of Political Economy*, 98(4), 703-738.
- Surowiecki, J. (2004). *The Wisdom of Crowds*. Doubleday.
- Tetlock, P. C. (2007). Giving content to investor sentiment: The role of media in the stock market. *The Journal of Finance*, 62(3), 1139-1168.

## Project Structure

```
VoxPopuli/
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
├── scripts/            # Utility scripts
└── tests/              # Test suite
```

## Dashboard

The React dashboard provides:
- Performance summary with key metrics
- Interactive equity curve
- Monthly returns heatmap
- Filterable trade explorer
- Strategy comparison overlay

## Configuration

Edit `config.yaml` to customize:

```yaml
data:
  subreddits: [wallstreetbets, stocks, investing]
  date_range:
    start: "2023-01-01"
    end: "2023-12-31"

strategy:
  name: attention_momentum
  params:
    top_n: 10
    holding_period: 5
    min_mentions: 5

backtest:
  initial_capital: 100000
  position_size: 0.1
  max_positions: 10
```

## Testing

```bash
pytest tests/
```

## Disclaimer

This project is for educational and research purposes only. The strategies and results presented do not constitute financial advice. Past performance does not guarantee future results. Always do your own research before making investment decisions.

## License

MIT License
