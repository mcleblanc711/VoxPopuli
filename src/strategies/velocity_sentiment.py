"""Velocity Sentiment Strategy - Trade on comment velocity spikes."""

from datetime import date

import pandas as pd

from .base_strategy import BaseStrategy, Signal, SignalType, Position


class VelocitySentimentStrategy(BaseStrategy):
    """Trade on comment velocity spikes combined with sentiment shifts.

    Strategy Logic:
    - Detect spikes in comment activity (velocity)
    - Confirm with sentiment shift in the same direction
    - Enter on breakout-like signals
    """

    name = "velocity_sentiment"
    description = "Trade on comment velocity spikes with sentiment"

    def __init__(
        self,
        velocity_threshold: float = 2.0,
        sentiment_shift_threshold: float = 0.2,
        min_mentions: int = 20,
        lookback_days: int = 7,
        spike_window: int = 1,
        holding_period: int = 3,
        **kwargs,
    ):
        """Initialize Velocity Sentiment strategy.

        Args:
            velocity_threshold: Multiplier over average for spike detection.
            sentiment_shift_threshold: Minimum sentiment change.
            min_mentions: Minimum mentions required.
            lookback_days: Days to calculate baseline.
            spike_window: Days to detect spike.
            holding_period: Days to hold positions.
            **kwargs: Base strategy parameters.
        """
        super().__init__(holding_period=holding_period, **kwargs)
        self.velocity_threshold = velocity_threshold
        self.sentiment_shift_threshold = sentiment_shift_threshold
        self.min_mentions = min_mentions
        self.lookback_days = lookback_days
        self.spike_window = spike_window

    def generate_signals(
        self,
        sentiment_data: pd.DataFrame,
        price_data: dict[str, pd.DataFrame],
        current_date: date,
        positions: list[Position],
    ) -> list[Signal]:
        """Generate signals based on velocity spikes.

        Args:
            sentiment_data: DataFrame with sentiment and comment data.
            price_data: Dict of ticker -> price DataFrame.
            current_date: Current date.
            positions: Current positions.

        Returns:
            List of signals.
        """
        signals = []

        # Get baseline data
        baseline_data = self.filter_sentiment_by_date(
            sentiment_data, current_date, self.lookback_days
        )

        # Get recent spike window data
        spike_data = self.filter_sentiment_by_date(
            sentiment_data, current_date, self.spike_window
        )

        if baseline_data.empty or spike_data.empty:
            return signals

        # Calculate velocity and sentiment metrics per ticker
        velocity_analysis = self._analyze_velocity(baseline_data, spike_data)

        if velocity_analysis.empty:
            return signals

        # Filter by velocity spike
        spikes = velocity_analysis[
            velocity_analysis["velocity_ratio"] >= self.velocity_threshold
        ]

        # Filter by sentiment shift
        spikes = spikes[
            spikes["sentiment_shift"].abs() >= self.sentiment_shift_threshold
        ]

        # Filter by minimum mentions
        spikes = spikes[spikes["recent_mentions"] >= self.min_mentions]

        # Sort by combined score
        spikes = spikes.sort_values("combined_score", ascending=False)

        held_tickers = {p.ticker for p in positions}
        available_slots = self.max_positions - len(positions)

        for _, row in spikes.iterrows():
            if available_slots <= 0:
                break

            ticker = row["ticker"]
            if ticker in held_tickers or ticker not in price_data:
                continue

            sentiment_shift = row["sentiment_shift"]

            # Determine direction based on sentiment shift
            if sentiment_shift > 0:
                signal_type = SignalType.LONG
                reason = f"Velocity spike ({row['velocity_ratio']:.1f}x) with positive sentiment shift (+{sentiment_shift:.2f})"
            elif sentiment_shift < 0 and self.allow_shorts:
                signal_type = SignalType.SHORT
                reason = f"Velocity spike ({row['velocity_ratio']:.1f}x) with negative sentiment shift ({sentiment_shift:.2f})"
            else:
                continue

            strength = min(1.0, row["combined_score"])

            signals.append(
                Signal(
                    date=current_date,
                    ticker=ticker,
                    signal_type=signal_type,
                    strength=strength,
                    reason=reason,
                    sentiment=row["recent_sentiment"],
                    mentions=int(row["recent_mentions"]),
                    metadata={
                        "velocity_ratio": row["velocity_ratio"],
                        "sentiment_shift": sentiment_shift,
                        "baseline_sentiment": row["baseline_sentiment"],
                        "comment_velocity": row.get("comment_velocity", 0),
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
                continue

            # Exit if velocity normalizes
            ticker_velocity = velocity_analysis[
                velocity_analysis["ticker"] == position.ticker
            ]

            if not ticker_velocity.empty:
                if ticker_velocity.iloc[0]["velocity_ratio"] < 1.2:
                    signals.append(
                        Signal(
                            date=current_date,
                            ticker=position.ticker,
                            signal_type=SignalType.CLOSE,
                            strength=1.0,
                            reason="velocity_normalized",
                        )
                    )

        return signals

    def _analyze_velocity(
        self,
        baseline_data: pd.DataFrame,
        spike_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Analyze velocity and sentiment for each ticker.

        Args:
            baseline_data: Baseline period data.
            spike_data: Recent spike window data.

        Returns:
            DataFrame with velocity analysis.
        """
        # Get baseline metrics per ticker
        baseline_metrics = self._get_ticker_metrics(baseline_data)

        # Get recent metrics per ticker
        recent_metrics = self._get_ticker_metrics(spike_data)

        if baseline_metrics.empty or recent_metrics.empty:
            return pd.DataFrame()

        # Merge and calculate velocity
        merged = recent_metrics.merge(
            baseline_metrics,
            on="ticker",
            suffixes=("_recent", "_baseline"),
        )

        if merged.empty:
            return pd.DataFrame()

        # Calculate velocity ratio
        merged["velocity_ratio"] = (
            merged["mentions_recent"] / (merged["mentions_baseline"] / self.lookback_days + 0.1)
        )

        # Calculate sentiment shift
        merged["sentiment_shift"] = (
            merged["sentiment_recent"] - merged["sentiment_baseline"]
        )

        # Calculate comment velocity if available
        if "comments_recent" in merged.columns and "comments_baseline" in merged.columns:
            merged["comment_velocity"] = (
                merged["comments_recent"] / (merged["comments_baseline"] / self.lookback_days + 0.1)
            )
        else:
            merged["comment_velocity"] = merged["velocity_ratio"]

        # Combined score
        merged["combined_score"] = (
            (merged["velocity_ratio"] / self.velocity_threshold) *
            (merged["sentiment_shift"].abs() / self.sentiment_shift_threshold) *
            0.5
        )

        # Rename for clarity
        merged = merged.rename(columns={
            "mentions_recent": "recent_mentions",
            "mentions_baseline": "baseline_mentions",
            "sentiment_recent": "recent_sentiment",
            "sentiment_baseline": "baseline_sentiment",
        })

        return merged

    def _get_ticker_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get aggregated metrics per ticker.

        Args:
            df: DataFrame with sentiment data.

        Returns:
            DataFrame with ticker metrics.
        """
        ticker_data: dict[str, dict] = {}

        for _, row in df.iterrows():
            tickers = row.get("tickers", [])
            sentiment = row.get("sentiment", 0)
            comments = row.get("num_comments", 0)

            if not isinstance(tickers, list):
                continue

            for ticker in tickers:
                if ticker not in ticker_data:
                    ticker_data[ticker] = {
                        "sentiments": [],
                        "mentions": 0,
                        "comments": 0,
                    }
                ticker_data[ticker]["sentiments"].append(sentiment)
                ticker_data[ticker]["mentions"] += 1
                ticker_data[ticker]["comments"] += comments

        results = []
        for ticker, data in ticker_data.items():
            if data["sentiments"]:
                results.append({
                    "ticker": ticker,
                    "sentiment": sum(data["sentiments"]) / len(data["sentiments"]),
                    "mentions": data["mentions"],
                    "comments": data["comments"],
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
