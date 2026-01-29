"""Contrarian Strategy - Fade extreme sentiment."""

from datetime import date

import pandas as pd

from .base_strategy import BaseStrategy, Signal, SignalType, Position


class ContrarianStrategy(BaseStrategy):
    """Fade extreme sentiment levels.

    Strategy Logic:
    - Short tickers with extreme positive sentiment (crowd is euphoric)
    - Long tickers with extreme negative sentiment (crowd is fearful)
    - Assumes mean reversion in sentiment
    """

    name = "contrarian"
    description = "Fade extreme sentiment"

    def __init__(
        self,
        extreme_threshold: float = 0.7,
        min_mentions: int = 30,
        lookback_days: int = 3,
        holding_period: int = 5,
        **kwargs,
    ):
        """Initialize Contrarian strategy.

        Args:
            extreme_threshold: Sentiment threshold for extreme (0.7 = top/bottom 30%).
            min_mentions: Minimum mentions required.
            lookback_days: Days to look back.
            holding_period: Days to hold positions.
            **kwargs: Base strategy parameters.
        """
        # Enable shorts by default for this strategy
        if "allow_shorts" not in kwargs:
            kwargs["allow_shorts"] = True

        super().__init__(holding_period=holding_period, **kwargs)
        self.extreme_threshold = extreme_threshold
        self.min_mentions = min_mentions
        self.lookback_days = lookback_days

    def generate_signals(
        self,
        sentiment_data: pd.DataFrame,
        price_data: dict[str, pd.DataFrame],
        current_date: date,
        positions: list[Position],
    ) -> list[Signal]:
        """Generate contrarian signals.

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

        # Aggregate sentiment per ticker
        ticker_stats = self._aggregate_ticker_stats(filtered_data)

        if ticker_stats.empty:
            return signals

        # Filter by minimum mentions
        ticker_stats = ticker_stats[ticker_stats["mentions"] >= self.min_mentions]

        if ticker_stats.empty:
            return signals

        # Find extreme sentiment tickers
        extreme_positive = ticker_stats[
            ticker_stats["sentiment"] >= self.extreme_threshold
        ].head(5)

        extreme_negative = ticker_stats[
            ticker_stats["sentiment"] <= -self.extreme_threshold
        ].head(5)

        held_tickers = {p.ticker for p in positions}
        available_slots = self.max_positions - len(positions)

        # Short extremely positive sentiment (if allowed)
        if self.allow_shorts:
            for _, row in extreme_positive.iterrows():
                if available_slots <= 0:
                    break

                ticker = row["ticker"]
                if ticker in held_tickers or ticker not in price_data:
                    continue

                strength = min(1.0, (row["sentiment"] - self.extreme_threshold) / 0.3 + 0.5)

                signals.append(
                    Signal(
                        date=current_date,
                        ticker=ticker,
                        signal_type=SignalType.SHORT,
                        strength=strength,
                        reason=f"Extreme positive sentiment ({row['sentiment']:.2f}), contrarian short",
                        sentiment=row["sentiment"],
                        mentions=int(row["mentions"]),
                    )
                )
                available_slots -= 1

        # Long extremely negative sentiment
        for _, row in extreme_negative.iterrows():
            if available_slots <= 0:
                break

            ticker = row["ticker"]
            if ticker in held_tickers or ticker not in price_data:
                continue

            strength = min(1.0, (abs(row["sentiment"]) - self.extreme_threshold) / 0.3 + 0.5)

            signals.append(
                Signal(
                    date=current_date,
                    ticker=ticker,
                    signal_type=SignalType.LONG,
                    strength=strength,
                    reason=f"Extreme negative sentiment ({row['sentiment']:.2f}), contrarian long",
                    sentiment=row["sentiment"],
                    mentions=int(row["mentions"]),
                )
            )
            available_slots -= 1

        # Check for exit signals
        for position in positions:
            current_price = self._get_current_price(
                price_data.get(position.ticker, pd.DataFrame()), current_date
            )

            should_exit, reason = self.should_exit(position, current_date, current_price)

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

            # Also exit if sentiment normalizes
            current_sentiment = self._get_ticker_sentiment(filtered_data, position.ticker)

            if current_sentiment is not None:
                if position.is_short and current_sentiment < self.extreme_threshold * 0.7:
                    signals.append(
                        Signal(
                            date=current_date,
                            ticker=position.ticker,
                            signal_type=SignalType.CLOSE,
                            strength=1.0,
                            reason="sentiment_normalized",
                        )
                    )
                elif position.is_long and current_sentiment > -self.extreme_threshold * 0.7:
                    signals.append(
                        Signal(
                            date=current_date,
                            ticker=position.ticker,
                            signal_type=SignalType.CLOSE,
                            strength=1.0,
                            reason="sentiment_normalized",
                        )
                    )

        return signals

    def _aggregate_ticker_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate sentiment per ticker."""
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

        if not result.empty:
            result = result.sort_values("sentiment", ascending=False)

        return result

    def _get_ticker_sentiment(self, df: pd.DataFrame, ticker: str) -> float | None:
        """Get current sentiment for a ticker."""
        sentiments = []
        for _, row in df.iterrows():
            tickers = row.get("tickers", [])
            if isinstance(tickers, list) and ticker in tickers:
                sentiments.append(row.get("sentiment", 0))

        return sum(sentiments) / len(sentiments) if sentiments else None

    def _get_current_price(self, price_df: pd.DataFrame, current_date: date) -> float:
        """Get current price."""
        if price_df.empty:
            return 0.0
        price_df = price_df[pd.to_datetime(price_df["date"]).dt.date <= current_date]
        if price_df.empty:
            return 0.0
        return float(price_df.iloc[-1]["adjusted_close"])
