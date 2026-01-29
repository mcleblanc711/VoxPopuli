"""Tests for trading strategies."""

import pytest
from datetime import date

import pandas as pd

from src.strategies.base_strategy import Signal, SignalType, Position
from src.strategies.attention_momentum import AttentionMomentumStrategy
from src.strategies.sentiment_divergence import SentimentDivergenceStrategy
from src.strategies.contrarian import ContrarianStrategy
from src.strategies.cross_subreddit import CrossSubredditStrategy
from src.strategies.velocity_sentiment import VelocitySentimentStrategy
from src.strategies import get_strategy, STRATEGY_REGISTRY


class TestSignal:
    """Tests for Signal dataclass."""

    def test_signal_creation(self):
        """Test creating a signal."""
        signal = Signal(
            date=date(2023, 1, 15),
            ticker="AAPL",
            signal_type=SignalType.LONG,
            strength=0.8,
            reason="Test signal",
            sentiment=0.5,
            mentions=100,
        )

        assert signal.ticker == "AAPL"
        assert signal.signal_type == SignalType.LONG
        assert signal.strength == 0.8

    def test_signal_to_dict(self):
        """Test signal serialization."""
        signal = Signal(
            date=date(2023, 1, 15),
            ticker="AAPL",
            signal_type=SignalType.LONG,
            strength=0.8,
            reason="Test",
        )

        d = signal.to_dict()

        assert d["ticker"] == "AAPL"
        assert d["signal_type"] == "long"
        assert d["date"] == "2023-01-15"


class TestPosition:
    """Tests for Position dataclass."""

    def test_position_creation(self):
        """Test creating a position."""
        position = Position(
            ticker="AAPL",
            entry_date=date(2023, 1, 15),
            entry_price=150.0,
            shares=100,
            direction="long",
        )

        assert position.ticker == "AAPL"
        assert position.is_long
        assert not position.is_short

    def test_position_pnl_long(self):
        """Test PnL calculation for long position."""
        position = Position(
            ticker="AAPL",
            entry_date=date(2023, 1, 15),
            entry_price=100.0,
            shares=10,
            direction="long",
        )

        # Price went up
        pnl = position.calculate_pnl(110.0)
        assert pnl == 100.0  # (110 - 100) * 10

    def test_position_pnl_short(self):
        """Test PnL calculation for short position."""
        position = Position(
            ticker="AAPL",
            entry_date=date(2023, 1, 15),
            entry_price=100.0,
            shares=10,
            direction="short",
        )

        # Price went down (good for short)
        pnl = position.calculate_pnl(90.0)
        assert pnl == 100.0  # (100 - 90) * 10

    def test_position_return(self):
        """Test return calculation."""
        position = Position(
            ticker="AAPL",
            entry_date=date(2023, 1, 15),
            entry_price=100.0,
            shares=10,
            direction="long",
        )

        ret = position.calculate_return(110.0)
        assert ret == 0.1  # 10% return


class TestAttentionMomentumStrategy:
    """Tests for Attention Momentum strategy."""

    @pytest.fixture
    def sample_sentiment_data(self):
        """Create sample sentiment data."""
        return pd.DataFrame({
            "created_utc": pd.to_datetime(["2023-01-15"] * 100),
            "tickers": [["AAPL"]] * 60 + [["TSLA"]] * 30 + [["GME"]] * 10,
            "sentiment": [0.5] * 60 + [0.3] * 30 + [0.1] * 10,
        })

    @pytest.fixture
    def sample_price_data(self):
        """Create sample price data."""
        dates = pd.date_range("2023-01-01", "2023-01-31", freq="D")

        return {
            "AAPL": pd.DataFrame({
                "date": dates,
                "adjusted_close": [150.0 + i * 0.5 for i in range(len(dates))],
            }),
            "TSLA": pd.DataFrame({
                "date": dates,
                "adjusted_close": [200.0 + i * 0.3 for i in range(len(dates))],
            }),
            "GME": pd.DataFrame({
                "date": dates,
                "adjusted_close": [25.0 + i * 0.1 for i in range(len(dates))],
            }),
        }

    def test_strategy_init(self):
        """Test strategy initialization."""
        strategy = AttentionMomentumStrategy(
            top_n=5,
            min_mentions=10,
            holding_period=5,
        )

        assert strategy.top_n == 5
        assert strategy.min_mentions == 10
        assert strategy.holding_period == 5

    def test_generate_signals(self, sample_sentiment_data, sample_price_data):
        """Test signal generation."""
        strategy = AttentionMomentumStrategy(
            top_n=2,
            min_mentions=5,
            holding_period=5,
        )

        signals = strategy.generate_signals(
            sample_sentiment_data,
            sample_price_data,
            date(2023, 1, 15),
            [],
        )

        # Should generate signals for AAPL and TSLA (top 2 by mentions)
        assert len(signals) == 2
        tickers = [s.ticker for s in signals]
        assert "AAPL" in tickers
        assert "TSLA" in tickers

    def test_generate_signals_min_mentions_filter(self, sample_sentiment_data, sample_price_data):
        """Test that min_mentions filter works."""
        strategy = AttentionMomentumStrategy(
            top_n=10,
            min_mentions=50,  # Only AAPL has >= 50 mentions
            holding_period=5,
        )

        signals = strategy.generate_signals(
            sample_sentiment_data,
            sample_price_data,
            date(2023, 1, 15),
            [],
        )

        assert len(signals) == 1
        assert signals[0].ticker == "AAPL"

    def test_generate_exit_signal(self, sample_sentiment_data, sample_price_data):
        """Test exit signal generation."""
        strategy = AttentionMomentumStrategy(
            top_n=2,
            min_mentions=5,
            holding_period=5,
        )

        # Create a position that's been held for 5+ days
        position = Position(
            ticker="AAPL",
            entry_date=date(2023, 1, 10),
            entry_price=150.0,
            shares=100,
            direction="long",
        )

        signals = strategy.generate_signals(
            sample_sentiment_data,
            sample_price_data,
            date(2023, 1, 16),  # 6 days after entry
            [position],
        )

        # Should have an exit signal
        exit_signals = [s for s in signals if s.signal_type == SignalType.CLOSE]
        assert len(exit_signals) == 1
        assert exit_signals[0].ticker == "AAPL"


class TestContrarianStrategy:
    """Tests for Contrarian strategy."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data with extreme sentiments."""
        return pd.DataFrame({
            "created_utc": pd.to_datetime(["2023-01-15"] * 100),
            "tickers": [["BULLISH"]] * 40 + [["BEARISH"]] * 40 + [["NEUTRAL"]] * 20,
            "sentiment": [0.9] * 40 + [-0.9] * 40 + [0.0] * 20,
        })

    @pytest.fixture
    def price_data(self):
        """Create price data."""
        dates = pd.date_range("2023-01-01", "2023-01-31", freq="D")
        return {
            "BULLISH": pd.DataFrame({"date": dates, "adjusted_close": [100.0] * len(dates)}),
            "BEARISH": pd.DataFrame({"date": dates, "adjusted_close": [50.0] * len(dates)}),
            "NEUTRAL": pd.DataFrame({"date": dates, "adjusted_close": [75.0] * len(dates)}),
        }

    def test_contrarian_long_on_negative(self, sample_data, price_data):
        """Test contrarian goes long on extreme negative sentiment."""
        strategy = ContrarianStrategy(
            extreme_threshold=0.7,
            min_mentions=10,
            allow_shorts=False,
        )

        signals = strategy.generate_signals(
            sample_data,
            price_data,
            date(2023, 1, 15),
            [],
        )

        # Should go long on BEARISH (extreme negative sentiment)
        long_signals = [s for s in signals if s.signal_type == SignalType.LONG]
        assert len(long_signals) == 1
        assert long_signals[0].ticker == "BEARISH"

    def test_contrarian_short_on_positive(self, sample_data, price_data):
        """Test contrarian goes short on extreme positive sentiment."""
        strategy = ContrarianStrategy(
            extreme_threshold=0.7,
            min_mentions=10,
            allow_shorts=True,
        )

        signals = strategy.generate_signals(
            sample_data,
            price_data,
            date(2023, 1, 15),
            [],
        )

        # Should go short on BULLISH (extreme positive sentiment)
        short_signals = [s for s in signals if s.signal_type == SignalType.SHORT]
        assert len(short_signals) == 1
        assert short_signals[0].ticker == "BULLISH"


class TestCrossSubredditStrategy:
    """Tests for Cross-Subreddit strategy."""

    @pytest.fixture
    def sample_data(self):
        """Create data with cross-subreddit mentions."""
        data = []
        # AAPL mentioned in 4 subreddits
        for sub in ["wsb", "stocks", "investing", "options"]:
            for _ in range(10):
                data.append({
                    "created_utc": pd.Timestamp("2023-01-15"),
                    "tickers": ["AAPL"],
                    "sentiment": 0.5,
                    "subreddit": sub,
                })
        # TSLA only in 2 subreddits
        for sub in ["wsb", "stocks"]:
            for _ in range(10):
                data.append({
                    "created_utc": pd.Timestamp("2023-01-15"),
                    "tickers": ["TSLA"],
                    "sentiment": 0.4,
                    "subreddit": sub,
                })

        return pd.DataFrame(data)

    @pytest.fixture
    def price_data(self):
        """Create price data."""
        dates = pd.date_range("2023-01-01", "2023-01-31", freq="D")
        return {
            "AAPL": pd.DataFrame({"date": dates, "adjusted_close": [150.0] * len(dates)}),
            "TSLA": pd.DataFrame({"date": dates, "adjusted_close": [200.0] * len(dates)}),
        }

    def test_cross_subreddit_consensus(self, sample_data, price_data):
        """Test that only tickers with enough subreddit consensus trigger signals."""
        strategy = CrossSubredditStrategy(
            min_subreddits=3,
            min_mentions_per_sub=5,
        )

        signals = strategy.generate_signals(
            sample_data,
            price_data,
            date(2023, 1, 15),
            [],
        )

        # Only AAPL has 3+ subreddits
        assert len(signals) == 1
        assert signals[0].ticker == "AAPL"


class TestStrategyRegistry:
    """Tests for strategy registry."""

    def test_get_strategy(self):
        """Test getting strategy by name."""
        strategy = get_strategy("attention_momentum", top_n=5)

        assert isinstance(strategy, AttentionMomentumStrategy)
        assert strategy.top_n == 5

    def test_get_strategy_invalid(self):
        """Test getting invalid strategy raises error."""
        with pytest.raises(ValueError, match="Unknown strategy"):
            get_strategy("nonexistent_strategy")

    def test_registry_contains_all_strategies(self):
        """Test that registry contains all strategies."""
        expected = [
            "attention_momentum",
            "sentiment_divergence",
            "contrarian",
            "cross_subreddit",
            "velocity_sentiment",
        ]

        for name in expected:
            assert name in STRATEGY_REGISTRY
