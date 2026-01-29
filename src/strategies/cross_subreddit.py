"""Cross-Subreddit Strategy - Require multi-subreddit consensus."""

from datetime import date

import pandas as pd

from .base_strategy import BaseStrategy, Signal, SignalType, Position


class CrossSubredditStrategy(BaseStrategy):
    """Trade when multiple subreddits agree on a ticker.

    Strategy Logic:
    - Count unique subreddits mentioning each ticker
    - Require minimum N subreddits to agree
    - Weight signal by consensus strength and sentiment alignment
    """

    name = "cross_subreddit"
    description = "Require multi-subreddit consensus"

    def __init__(
        self,
        min_subreddits: int = 3,
        min_mentions_per_sub: int = 5,
        sentiment_alignment_threshold: float = 0.2,
        lookback_days: int = 3,
        holding_period: int = 7,
        **kwargs,
    ):
        """Initialize Cross-Subreddit strategy.

        Args:
            min_subreddits: Minimum subreddits for consensus.
            min_mentions_per_sub: Minimum mentions per subreddit.
            sentiment_alignment_threshold: Std dev threshold for alignment.
            lookback_days: Days to look back.
            holding_period: Days to hold positions.
            **kwargs: Base strategy parameters.
        """
        super().__init__(holding_period=holding_period, **kwargs)
        self.min_subreddits = min_subreddits
        self.min_mentions_per_sub = min_mentions_per_sub
        self.sentiment_alignment_threshold = sentiment_alignment_threshold
        self.lookback_days = lookback_days

    def generate_signals(
        self,
        sentiment_data: pd.DataFrame,
        price_data: dict[str, pd.DataFrame],
        current_date: date,
        positions: list[Position],
    ) -> list[Signal]:
        """Generate signals based on cross-subreddit consensus.

        Args:
            sentiment_data: DataFrame with 'subreddit' column.
            price_data: Dict of ticker -> price DataFrame.
            current_date: Current date.
            positions: Current positions.

        Returns:
            List of signals.
        """
        signals = []

        # Filter to lookback period
        filtered_data = self.filter_sentiment_by_date(
            sentiment_data, current_date, self.lookback_days
        )

        if filtered_data.empty:
            return signals

        # Analyze cross-subreddit consensus
        consensus = self._analyze_consensus(filtered_data)

        if consensus.empty:
            return signals

        # Filter by minimum subreddits
        consensus = consensus[consensus["subreddit_count"] >= self.min_subreddits]

        # Sort by consensus score (subreddit count * sentiment consistency)
        consensus = consensus.sort_values("consensus_score", ascending=False)

        held_tickers = {p.ticker for p in positions}
        available_slots = self.max_positions - len(positions)

        for _, row in consensus.iterrows():
            if available_slots <= 0:
                break

            ticker = row["ticker"]
            if ticker in held_tickers or ticker not in price_data:
                continue

            avg_sentiment = row["avg_sentiment"]
            sentiment_std = row["sentiment_std"]

            # Check sentiment alignment (low std = high agreement)
            if sentiment_std > self.sentiment_alignment_threshold:
                continue

            # Determine direction based on average sentiment
            if avg_sentiment > 0.1:
                signal_type = SignalType.LONG
                reason = f"Cross-sub bullish consensus: {row['subreddit_count']} subs, sentiment={avg_sentiment:.2f}"
            elif avg_sentiment < -0.1 and self.allow_shorts:
                signal_type = SignalType.SHORT
                reason = f"Cross-sub bearish consensus: {row['subreddit_count']} subs, sentiment={avg_sentiment:.2f}"
            else:
                continue

            # Signal strength based on consensus quality
            strength = min(1.0, row["consensus_score"])

            signals.append(
                Signal(
                    date=current_date,
                    ticker=ticker,
                    signal_type=signal_type,
                    strength=strength,
                    reason=reason,
                    sentiment=avg_sentiment,
                    mentions=int(row["total_mentions"]),
                    metadata={
                        "subreddit_count": row["subreddit_count"],
                        "subreddits": row["subreddits"],
                        "sentiment_std": sentiment_std,
                    },
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

        return signals

    def _analyze_consensus(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze cross-subreddit consensus for each ticker.

        Args:
            df: DataFrame with 'tickers', 'subreddit', 'sentiment' columns.

        Returns:
            DataFrame with consensus analysis.
        """
        # Group by ticker and subreddit
        ticker_sub_data: dict[str, dict[str, list]] = {}

        for _, row in df.iterrows():
            tickers = row.get("tickers", [])
            subreddit = row.get("subreddit", "unknown")
            sentiment = row.get("sentiment", 0)

            if not isinstance(tickers, list):
                continue

            for ticker in tickers:
                if ticker not in ticker_sub_data:
                    ticker_sub_data[ticker] = {}

                if subreddit not in ticker_sub_data[ticker]:
                    ticker_sub_data[ticker][subreddit] = []

                ticker_sub_data[ticker][subreddit].append(sentiment)

        # Calculate consensus metrics
        results = []

        for ticker, sub_data in ticker_sub_data.items():
            # Filter subreddits with minimum mentions
            valid_subs = {
                sub: sents
                for sub, sents in sub_data.items()
                if len(sents) >= self.min_mentions_per_sub
            }

            if len(valid_subs) < self.min_subreddits:
                continue

            # Calculate per-subreddit sentiment
            sub_sentiments = []
            total_mentions = 0

            for sub, sents in valid_subs.items():
                sub_sentiments.append(sum(sents) / len(sents))
                total_mentions += len(sents)

            avg_sentiment = sum(sub_sentiments) / len(sub_sentiments)

            # Calculate sentiment standard deviation (alignment)
            if len(sub_sentiments) > 1:
                mean_sent = sum(sub_sentiments) / len(sub_sentiments)
                variance = sum((s - mean_sent) ** 2 for s in sub_sentiments) / len(sub_sentiments)
                sentiment_std = variance ** 0.5
            else:
                sentiment_std = 0

            # Consensus score: more subs + lower variance = higher score
            alignment_factor = max(0, 1 - sentiment_std / 0.5)
            consensus_score = (len(valid_subs) / 5) * alignment_factor * abs(avg_sentiment)

            results.append({
                "ticker": ticker,
                "subreddit_count": len(valid_subs),
                "subreddits": list(valid_subs.keys()),
                "total_mentions": total_mentions,
                "avg_sentiment": avg_sentiment,
                "sentiment_std": sentiment_std,
                "consensus_score": consensus_score,
            })

        return pd.DataFrame(results)

    def _get_current_price(self, price_df: pd.DataFrame, current_date: date) -> float:
        """Get current price."""
        if price_df.empty:
            return 0.0
        price_df = price_df[pd.to_datetime(price_df["date"]).dt.date <= current_date]
        if price_df.empty:
            return 0.0
        return float(price_df.iloc[-1]["adjusted_close"])
