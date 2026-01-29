"""Attention Momentum Strategy - Long top N tickers by mention count."""

from datetime import date

import pandas as pd

from .base_strategy import BaseStrategy, Signal, SignalType, Position


class AttentionMomentumStrategy(BaseStrategy):
    """Long the most-mentioned tickers.

    Strategy Logic:
    - Count ticker mentions over lookback period
    - Long top N tickers by mention count
    - Require minimum mention threshold
    - Hold for fixed period or until exit signal
    """

    name = "attention_momentum"
    description = "Long top N tickers by mention count"

    def __init__(
        self,
        top_n: int = 10,
        min_mentions: int = 50,
        lookback_days: int = 1,
        holding_period: int = 5,
        min_sentiment: float | None = None,
        **kwargs,
    ):
        """Initialize Attention Momentum strategy.

        Args:
            top_n: Number of top tickers to long.
            min_mentions: Minimum mentions required.
            lookback_days: Days to look back for mentions.
            holding_period: Days to hold positions.
            min_sentiment: Optional minimum sentiment filter.
            **kwargs: Base strategy parameters.
        """
        super().__init__(holding_period=holding_period, **kwargs)
        self.top_n = top_n
        self.min_mentions = min_mentions
        self.lookback_days = lookback_days
        self.min_sentiment = min_sentiment

    def generate_signals(
        self,
        sentiment_data: pd.DataFrame,
        price_data: dict[str, pd.DataFrame],
        current_date: date,
        positions: list[Position],
    ) -> list[Signal]:
        """Generate signals based on mention counts.

        Args:
            sentiment_data: DataFrame with 'tickers' and 'sentiment' columns.
            price_data: Dict of ticker -> price DataFrame.
            current_date: Current date.
            positions: Current positions.

        Returns:
            List of signals.
        """
        signals = []

        # Filter data to lookback period
        filtered_data = self.filter_sentiment_by_date(
            sentiment_data, current_date, self.lookback_days
        )

        if filtered_data.empty:
            return signals

        # Count mentions
        ticker_counts = self._count_mentions(filtered_data)

        if ticker_counts.empty:
            return signals

        # Filter by minimum mentions
        ticker_counts = ticker_counts[ticker_counts["count"] >= self.min_mentions]

        # Get sentiment for each ticker if available
        ticker_sentiment = self._get_ticker_sentiment(filtered_data)

        # Apply minimum sentiment filter if set
        if self.min_sentiment is not None:
            valid_tickers = ticker_sentiment[
                ticker_sentiment["sentiment"] >= self.min_sentiment
            ]["ticker"].tolist()
            ticker_counts = ticker_counts[ticker_counts["ticker"].isin(valid_tickers)]

        # Get top N
        top_tickers = ticker_counts.head(self.top_n)

        # Get currently held tickers
        held_tickers = {p.ticker for p in positions}

        # Generate long signals for new positions
        available_slots = self.max_positions - len(positions)

        for _, row in top_tickers.iterrows():
            if available_slots <= 0:
                break

            ticker = row["ticker"]

            # Skip if already holding
            if ticker in held_tickers:
                continue

            # Skip if no price data
            if ticker not in price_data:
                continue

            # Calculate signal strength based on relative mentions
            max_mentions = top_tickers["count"].max()
            strength = row["count"] / max_mentions if max_mentions > 0 else 0.5

            sentiment = ticker_sentiment[ticker_sentiment["ticker"] == ticker]
            sentiment_val = sentiment["sentiment"].iloc[0] if not sentiment.empty else None

            signals.append(
                Signal(
                    date=current_date,
                    ticker=ticker,
                    signal_type=SignalType.LONG,
                    strength=strength,
                    reason=f"Top {self.top_n} by mentions ({row['count']} mentions)",
                    sentiment=sentiment_val,
                    mentions=int(row["count"]),
                )
            )

            available_slots -= 1

        # Check for exit signals on existing positions
        for position in positions:
            # Check holding period
            should_exit, reason = self.should_exit(
                position,
                current_date,
                self._get_current_price(price_data, position.ticker, current_date),
            )

            if should_exit:
                signals.append(
                    Signal(
                        date=current_date,
                        ticker=position.ticker,
                        signal_type=SignalType.CLOSE,
                        strength=1.0,
                        reason=reason,
                    )
                )

        return signals

    def _count_mentions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Count ticker mentions in the data.

        Args:
            df: DataFrame with 'tickers' column (list of tickers per row).

        Returns:
            DataFrame with ticker and count columns.
        """
        ticker_counts: dict[str, int] = {}

        for tickers in df.get("tickers", []):
            if isinstance(tickers, list):
                for ticker in tickers:
                    ticker_counts[ticker] = ticker_counts.get(ticker, 0) + 1

        result = pd.DataFrame([
            {"ticker": t, "count": c}
            for t, c in ticker_counts.items()
        ])

        if not result.empty:
            result = result.sort_values("count", ascending=False)

        return result

    def _get_ticker_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get average sentiment per ticker.

        Args:
            df: DataFrame with 'tickers' and 'sentiment' columns.

        Returns:
            DataFrame with ticker and sentiment columns.
        """
        ticker_sentiments: dict[str, list[float]] = {}

        for _, row in df.iterrows():
            tickers = row.get("tickers", [])
            sentiment = row.get("sentiment", 0)

            if isinstance(tickers, list):
                for ticker in tickers:
                    if ticker not in ticker_sentiments:
                        ticker_sentiments[ticker] = []
                    ticker_sentiments[ticker].append(sentiment)

        result = pd.DataFrame([
            {"ticker": t, "sentiment": sum(s) / len(s)}
            for t, s in ticker_sentiments.items()
            if s
        ])

        return result

    def _get_current_price(
        self,
        price_data: dict[str, pd.DataFrame],
        ticker: str,
        current_date: date,
    ) -> float:
        """Get current price for a ticker.

        Args:
            price_data: Dict of price DataFrames.
            ticker: Ticker symbol.
            current_date: Current date.

        Returns:
            Current price or 0 if not found.
        """
        if ticker not in price_data:
            return 0.0

        df = price_data[ticker]
        df = df[pd.to_datetime(df["date"]).dt.date <= current_date]

        if df.empty:
            return 0.0

        return float(df.iloc[-1]["adjusted_close"])
