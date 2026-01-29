"""Sentiment Divergence Strategy - Trade sentiment/price divergences."""

from datetime import date

import pandas as pd

from .base_strategy import BaseStrategy, Signal, SignalType, Position


class SentimentDivergenceStrategy(BaseStrategy):
    """Trade divergences between sentiment and price.

    Strategy Logic:
    - Long: Positive sentiment but flat/negative price (undervalued)
    - Short: Negative sentiment but extended price (overvalued)
    - Exit when divergence resolves
    """

    name = "sentiment_divergence"
    description = "Trade sentiment/price divergences"

    def __init__(
        self,
        sentiment_threshold: float = 0.3,
        price_change_threshold: float = 0.02,
        lookback_days: int = 5,
        min_mentions: int = 20,
        holding_period: int = 10,
        **kwargs,
    ):
        """Initialize Sentiment Divergence strategy.

        Args:
            sentiment_threshold: Minimum |sentiment| for signal.
            price_change_threshold: Price change threshold (e.g., 0.02 = 2%).
            lookback_days: Days to look back for price change.
            min_mentions: Minimum mentions required.
            holding_period: Days to hold positions.
            **kwargs: Base strategy parameters.
        """
        super().__init__(holding_period=holding_period, **kwargs)
        self.sentiment_threshold = sentiment_threshold
        self.price_change_threshold = price_change_threshold
        self.lookback_days = lookback_days
        self.min_mentions = min_mentions

    def generate_signals(
        self,
        sentiment_data: pd.DataFrame,
        price_data: dict[str, pd.DataFrame],
        current_date: date,
        positions: list[Position],
    ) -> list[Signal]:
        """Generate signals based on sentiment/price divergence.

        Args:
            sentiment_data: DataFrame with sentiment data.
            price_data: Dict of ticker -> price DataFrame.
            current_date: Current date.
            positions: Current positions.

        Returns:
            List of signals.
        """
        signals = []

        # Get recent sentiment
        filtered_data = self.filter_sentiment_by_date(
            sentiment_data, current_date, self.lookback_days
        )

        if filtered_data.empty:
            return signals

        # Aggregate sentiment and mentions per ticker
        ticker_stats = self._aggregate_ticker_stats(filtered_data)

        if ticker_stats.empty:
            return signals

        # Filter by minimum mentions
        ticker_stats = ticker_stats[ticker_stats["mentions"] >= self.min_mentions]

        # Get held tickers
        held_tickers = {p.ticker for p in positions}
        available_slots = self.max_positions - len(positions)

        for _, row in ticker_stats.iterrows():
            if available_slots <= 0:
                break

            ticker = row["ticker"]
            sentiment = row["sentiment"]

            # Skip if already holding
            if ticker in held_tickers:
                continue

            # Skip if no price data
            if ticker not in price_data:
                continue

            # Calculate price change
            price_change = self._calculate_price_change(
                price_data[ticker], current_date, self.lookback_days
            )

            if price_change is None:
                continue

            # Check for divergence
            signal = self._check_divergence(
                ticker, sentiment, price_change, row["mentions"], current_date
            )

            if signal:
                signals.append(signal)
                available_slots -= 1

        # Check for exit signals
        for position in positions:
            if position.ticker not in price_data:
                continue

            current_price = self._get_current_price(
                price_data[position.ticker], current_date
            )

            should_exit, reason = self.should_exit(
                position, current_date, current_price
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
                continue

            # Check if divergence has resolved
            recent_data = self.filter_sentiment_by_date(
                sentiment_data, current_date, 1
            )
            current_sentiment = self._get_ticker_sentiment(recent_data, position.ticker)
            price_change = self._calculate_price_change(
                price_data[position.ticker], current_date, 3
            )

            if self._divergence_resolved(position, current_sentiment, price_change):
                signals.append(
                    Signal(
                        date=current_date,
                        ticker=position.ticker,
                        signal_type=SignalType.CLOSE,
                        strength=1.0,
                        reason="divergence_resolved",
                    )
                )

        return signals

    def _aggregate_ticker_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate sentiment and mentions per ticker.

        Args:
            df: DataFrame with 'tickers' and 'sentiment' columns.

        Returns:
            DataFrame with ticker, sentiment, mentions columns.
        """
        ticker_data: dict[str, dict] = {}

        for _, row in df.iterrows():
            tickers = row.get("tickers", [])
            sentiment = row.get("sentiment", 0)

            if isinstance(tickers, list):
                for ticker in tickers:
                    if ticker not in ticker_data:
                        ticker_data[ticker] = {"sentiments": [], "mentions": 0}
                    ticker_data[ticker]["sentiments"].append(sentiment)
                    ticker_data[ticker]["mentions"] += 1

        result = pd.DataFrame([
            {
                "ticker": t,
                "sentiment": sum(d["sentiments"]) / len(d["sentiments"]),
                "mentions": d["mentions"],
            }
            for t, d in ticker_data.items()
            if d["sentiments"]
        ])

        return result

    def _calculate_price_change(
        self,
        price_df: pd.DataFrame,
        current_date: date,
        lookback_days: int,
    ) -> float | None:
        """Calculate price change over lookback period.

        Args:
            price_df: Price DataFrame.
            current_date: Current date.
            lookback_days: Days to look back.

        Returns:
            Price change as decimal or None.
        """
        price_df = price_df.copy()
        price_df["date"] = pd.to_datetime(price_df["date"]).dt.date

        # Get current and past prices
        current = price_df[price_df["date"] <= current_date].tail(1)
        if current.empty:
            return None

        past_date = pd.Timestamp(current_date) - pd.Timedelta(days=lookback_days)
        past = price_df[price_df["date"] <= past_date.date()].tail(1)

        if past.empty:
            return None

        current_price = current["adjusted_close"].iloc[0]
        past_price = past["adjusted_close"].iloc[0]

        return (current_price - past_price) / past_price

    def _check_divergence(
        self,
        ticker: str,
        sentiment: float,
        price_change: float,
        mentions: int,
        current_date: date,
    ) -> Signal | None:
        """Check for sentiment/price divergence.

        Args:
            ticker: Ticker symbol.
            sentiment: Current sentiment.
            price_change: Recent price change.
            mentions: Number of mentions.
            current_date: Current date.

        Returns:
            Signal if divergence found, None otherwise.
        """
        # Bullish divergence: positive sentiment, flat/negative price
        if sentiment >= self.sentiment_threshold and price_change <= -self.price_change_threshold:
            strength = min(1.0, abs(sentiment) * 2)  # Scale strength
            return Signal(
                date=current_date,
                ticker=ticker,
                signal_type=SignalType.LONG,
                strength=strength,
                reason=f"Bullish divergence: sentiment={sentiment:.2f}, price={price_change:.2%}",
                sentiment=sentiment,
                mentions=mentions,
                metadata={"price_change": price_change},
            )

        # Bearish divergence: negative sentiment, positive price
        if self.allow_shorts:
            if sentiment <= -self.sentiment_threshold and price_change >= self.price_change_threshold:
                strength = min(1.0, abs(sentiment) * 2)
                return Signal(
                    date=current_date,
                    ticker=ticker,
                    signal_type=SignalType.SHORT,
                    strength=strength,
                    reason=f"Bearish divergence: sentiment={sentiment:.2f}, price={price_change:.2%}",
                    sentiment=sentiment,
                    mentions=mentions,
                    metadata={"price_change": price_change},
                )

        return None

    def _get_ticker_sentiment(self, df: pd.DataFrame, ticker: str) -> float | None:
        """Get sentiment for a specific ticker.

        Args:
            df: Sentiment DataFrame.
            ticker: Ticker to look up.

        Returns:
            Sentiment or None.
        """
        for _, row in df.iterrows():
            tickers = row.get("tickers", [])
            if isinstance(tickers, list) and ticker in tickers:
                return row.get("sentiment", 0)
        return None

    def _divergence_resolved(
        self,
        position: Position,
        current_sentiment: float | None,
        price_change: float | None,
    ) -> bool:
        """Check if the divergence has resolved.

        Args:
            position: Current position.
            current_sentiment: Current sentiment.
            price_change: Recent price change.

        Returns:
            True if divergence resolved.
        """
        if current_sentiment is None or price_change is None:
            return False

        # For long positions (bullish divergence), resolved when price catches up
        if position.is_long and price_change >= self.price_change_threshold:
            return True

        # For short positions, resolved when price drops
        if position.is_short and price_change <= -self.price_change_threshold:
            return True

        return False

    def _get_current_price(self, price_df: pd.DataFrame, current_date: date) -> float:
        """Get current price."""
        price_df = price_df[pd.to_datetime(price_df["date"]).dt.date <= current_date]
        if price_df.empty:
            return 0.0
        return float(price_df.iloc[-1]["adjusted_close"])
