# VoxPopuli Scratchpad

## Ideas & Notes

### Data Sources
- Arctic Shift is the main source for Reddit historical data
- Pushshift API is deprecated, use dumps instead
- Consider rate limiting for yfinance to avoid bans

### Strategy Ideas
- Could add earnings calendar filter to avoid holding through earnings
- Consider adding options flow data as additional signal
- VIX filter to reduce exposure during high volatility

### Performance Optimizations
- Use polars instead of pandas for faster processing
- Vectorized backtest operations are critical for speed
- Consider dask for parallel processing of large Reddit dumps

### Dashboard Enhancements
- Dark mode toggle
- Export to PDF report
- Shareable links with encoded parameters

### False Positive Tickers to Filter
```
A, I, CEO, DD, YOLO, FOMO, ATH, ATL, EOD, EOM, EOY,
IPO, FDA, SEC, ETF, ITM, OTM, ATM, IV, DTE, OP,
WSB, HODL, FD, LEAPS, PUTS, CALLS, RH, TDA, IBKR,
PM, AM, PT, DD, TA, FA, GDP, CPI, FOMC, JPM
```

### Subreddits to Monitor
Primary:
- wallstreetbets (highest volume, most noise)
- stocks (more balanced)
- investing (conservative bias)
- options (derivatives focus)
- stockmarket (general)

Secondary:
- SPACs
- pennystocks
- thetagang
- dividends
- ValueInvesting

### Sentiment Calibration Notes
- WSB tends hyperbolic, may need sentiment dampening
- r/investing tends conservative, may need amplification
- Consider subreddit-specific calibration factors

### Backtest Gotchas
- Survivorship bias in ticker lists
- Look-ahead bias in sentiment aggregation
- Weekend/holiday handling for trade execution
- Market hours vs after-hours pricing
