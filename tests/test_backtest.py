"""Tests for backtest engine."""

import pytest
from datetime import date

import pandas as pd
import numpy as np

from src.backtest.engine import BacktestEngine, Trade, run_backtest
from src.backtest.metrics import (
    MetricsCalculator,
    PerformanceMetrics,
    calculate_metrics,
    calculate_monthly_returns,
)
from src.backtest.export import BacktestExporter
from src.strategies.attention_momentum import AttentionMomentumStrategy


class TestMetricsCalculator:
    """Tests for MetricsCalculator."""

    @pytest.fixture
    def sample_equity_curve(self):
        """Create sample equity curve."""
        dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")
        # Simulate growth with some volatility
        np.random.seed(42)
        returns = np.random.normal(0.0005, 0.02, len(dates))
        values = [100000]
        for r in returns:
            values.append(values[-1] * (1 + r))
        return pd.Series(values[1:], index=dates)

    @pytest.fixture
    def sample_trades(self):
        """Create sample trades DataFrame."""
        return pd.DataFrame({
            "ticker": ["AAPL", "TSLA", "GME", "AMD", "NVDA"],
            "pnl": [500, -200, 1000, -300, 800],
            "entry_date": ["2023-01-10", "2023-02-15", "2023-03-20", "2023-05-01", "2023-06-15"],
            "exit_date": ["2023-01-15", "2023-02-20", "2023-03-25", "2023-05-06", "2023-06-20"],
        })

    def test_calculate_returns(self, sample_equity_curve):
        """Test return calculation."""
        calc = MetricsCalculator()
        returns = calc.calculate_returns(sample_equity_curve)

        assert len(returns) == len(sample_equity_curve) - 1
        # Returns should be reasonable
        assert returns.abs().max() < 0.5  # No single day > 50%

    def test_calculate_total_return(self, sample_equity_curve):
        """Test total return calculation."""
        calc = MetricsCalculator()
        total_return = calc.calculate_total_return(sample_equity_curve)

        # Should be close to (final / initial) - 1
        expected = (sample_equity_curve.iloc[-1] / sample_equity_curve.iloc[0]) - 1
        assert abs(total_return - expected) < 0.001

    def test_calculate_sharpe_ratio(self, sample_equity_curve):
        """Test Sharpe ratio calculation."""
        calc = MetricsCalculator()
        returns = calc.calculate_returns(sample_equity_curve)
        sharpe = calc.calculate_sharpe_ratio(returns)

        # Sharpe should be reasonable
        assert -5 < sharpe < 5

    def test_calculate_sortino_ratio(self, sample_equity_curve):
        """Test Sortino ratio calculation."""
        calc = MetricsCalculator()
        returns = calc.calculate_returns(sample_equity_curve)
        sortino = calc.calculate_sortino_ratio(returns)

        # Sortino should be >= Sharpe (less penalty for upside vol)
        sharpe = calc.calculate_sharpe_ratio(returns)
        # Allow for some numerical variance
        assert sortino >= sharpe * 0.8

    def test_calculate_max_drawdown(self, sample_equity_curve):
        """Test max drawdown calculation."""
        calc = MetricsCalculator()
        max_dd, duration = calc.calculate_max_drawdown(sample_equity_curve)

        # Drawdown should be negative
        assert max_dd <= 0
        # Duration should be positive
        assert duration >= 0

    def test_calculate_win_rate(self, sample_trades):
        """Test win rate calculation."""
        calc = MetricsCalculator()
        win_rate = calc.calculate_win_rate(sample_trades)

        # 3 winners out of 5
        assert win_rate == 0.6

    def test_calculate_profit_factor(self, sample_trades):
        """Test profit factor calculation."""
        calc = MetricsCalculator()
        profit_factor = calc.calculate_profit_factor(sample_trades)

        # Gross profit = 500 + 1000 + 800 = 2300
        # Gross loss = 200 + 300 = 500
        expected = 2300 / 500
        assert abs(profit_factor - expected) < 0.001

    def test_calculate_trade_stats(self, sample_trades):
        """Test trade statistics calculation."""
        calc = MetricsCalculator()
        stats = calc.calculate_trade_stats(sample_trades)

        assert stats["total_trades"] == 5
        assert stats["winning_trades"] == 3
        assert stats["losing_trades"] == 2

    def test_calculate_all_metrics(self, sample_equity_curve, sample_trades):
        """Test complete metrics calculation."""
        calc = MetricsCalculator()
        metrics = calc.calculate_all_metrics(sample_equity_curve, sample_trades)

        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.total_trades == 5
        assert 0 <= metrics.win_rate <= 1

    def test_metrics_to_dict(self, sample_equity_curve, sample_trades):
        """Test metrics serialization."""
        calc = MetricsCalculator()
        metrics = calc.calculate_all_metrics(sample_equity_curve, sample_trades)

        d = metrics.to_dict()

        assert "total_return" in d
        assert "sharpe_ratio" in d
        assert "max_drawdown" in d


class TestCalculateMonthlyReturns:
    """Tests for monthly return calculation."""

    def test_calculate_monthly_returns(self):
        """Test monthly return calculation."""
        # Create equity curve spanning multiple months
        dates = pd.date_range("2023-01-01", "2023-03-31", freq="D")
        values = [100000 + i * 100 for i in range(len(dates))]
        equity_curve = pd.Series(values, index=dates)

        monthly = calculate_monthly_returns(equity_curve)

        assert "year" in monthly.columns
        assert "month" in monthly.columns
        assert "return" in monthly.columns
        assert len(monthly) >= 2  # At least Feb and Mar


class TestBacktestEngine:
    """Tests for BacktestEngine."""

    @pytest.fixture
    def simple_sentiment_data(self):
        """Create simple sentiment data."""
        dates = pd.date_range("2023-01-01", "2023-01-31", freq="D")
        data = []
        for d in dates:
            data.append({
                "created_utc": d,
                "tickers": ["AAPL"],
                "sentiment": 0.5,
            })
        return pd.DataFrame(data)

    @pytest.fixture
    def simple_price_data(self):
        """Create simple price data with uptrend."""
        dates = pd.date_range("2023-01-01", "2023-01-31", freq="D")
        return {
            "AAPL": pd.DataFrame({
                "date": dates,
                "adjusted_close": [150.0 + i for i in range(len(dates))],
            }),
            "SPY": pd.DataFrame({
                "date": dates,
                "adjusted_close": [400.0 + i * 0.5 for i in range(len(dates))],
            }),
        }

    def test_engine_init(self):
        """Test engine initialization."""
        engine = BacktestEngine(initial_capital=50000)

        assert engine.initial_capital == 50000

    def test_run_backtest(self, simple_sentiment_data, simple_price_data):
        """Test running a backtest."""
        engine = BacktestEngine(initial_capital=100000)
        strategy = AttentionMomentumStrategy(
            top_n=1,
            min_mentions=1,
            holding_period=5,
        )

        result = engine.run(
            strategy=strategy,
            sentiment_data=simple_sentiment_data,
            price_data=simple_price_data,
            start_date="2023-01-01",
            end_date="2023-01-31",
        )

        assert result.strategy_name == "attention_momentum"
        assert result.initial_capital == 100000
        assert len(result.equity_curve) > 0

    def test_run_backtest_with_benchmark(self, simple_sentiment_data, simple_price_data):
        """Test backtest with benchmark comparison."""
        engine = BacktestEngine()
        strategy = AttentionMomentumStrategy(
            top_n=1,
            min_mentions=1,
            holding_period=5,
        )

        result = engine.run(
            strategy=strategy,
            sentiment_data=simple_sentiment_data,
            price_data=simple_price_data,
            start_date="2023-01-01",
            end_date="2023-01-31",
            benchmark_ticker="SPY",
        )

        # Benchmark metrics should be populated
        assert result.metrics.benchmark_return is not None


class TestTrade:
    """Tests for Trade dataclass."""

    def test_trade_creation(self):
        """Test creating a trade."""
        trade = Trade(
            ticker="AAPL",
            direction="long",
            entry_date=date(2023, 1, 10),
            exit_date=date(2023, 1, 15),
            entry_price=150.0,
            exit_price=155.0,
            shares=100,
            pnl=500.0,
            return_pct=0.0333,
            exit_reason="holding_period",
        )

        assert trade.ticker == "AAPL"
        assert trade.pnl == 500.0

    def test_trade_to_dict(self):
        """Test trade serialization."""
        trade = Trade(
            ticker="AAPL",
            direction="long",
            entry_date=date(2023, 1, 10),
            exit_date=date(2023, 1, 15),
            entry_price=150.0,
            exit_price=155.0,
            shares=100,
            pnl=500.0,
            return_pct=0.0333,
        )

        d = trade.to_dict()

        assert d["ticker"] == "AAPL"
        assert d["entry_date"] == "2023-01-10"


class TestBacktestExporter:
    """Tests for BacktestExporter."""

    @pytest.fixture
    def sample_results(self):
        """Create sample results for export."""
        dates = pd.date_range("2023-01-01", "2023-01-31", freq="D")
        equity_curve = pd.Series([100000 + i * 100 for i in range(len(dates))], index=dates)

        trades = pd.DataFrame({
            "ticker": ["AAPL", "TSLA"],
            "direction": ["long", "long"],
            "entry_date": ["2023-01-05", "2023-01-15"],
            "exit_date": ["2023-01-10", "2023-01-20"],
            "entry_price": [150.0, 200.0],
            "exit_price": [155.0, 195.0],
            "shares": [100, 50],
            "pnl": [500.0, -250.0],
            "return_pct": [0.033, -0.025],
        })

        metrics = PerformanceMetrics(
            total_return=0.03,
            cagr=0.36,
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            max_drawdown=-0.05,
            max_drawdown_duration=5,
            win_rate=0.5,
            profit_factor=2.0,
            avg_win=500.0,
            avg_loss=-250.0,
            total_trades=2,
            winning_trades=1,
            losing_trades=1,
            avg_holding_period=5.0,
            exposure_time=0.8,
            volatility=0.15,
            calmar_ratio=7.2,
        )

        return {
            "metrics": metrics,
            "equity_curve": equity_curve,
            "trades": trades,
            "config": {"strategy": "test"},
        }

    def test_prepare_equity_curve(self, sample_results):
        """Test equity curve preparation."""
        exporter = BacktestExporter()
        prepared = exporter.prepare_equity_curve(sample_results["equity_curve"])

        assert len(prepared) > 0
        assert "date" in prepared[0]
        assert "value" in prepared[0]

    def test_prepare_monthly_heatmap(self, sample_results):
        """Test monthly heatmap preparation."""
        exporter = BacktestExporter()
        heatmap = exporter.prepare_monthly_heatmap(sample_results["equity_curve"])

        assert "2023" in heatmap
        assert isinstance(heatmap["2023"], dict)


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_calculate_metrics_function(self):
        """Test calculate_metrics convenience function."""
        dates = pd.date_range("2023-01-01", "2023-01-31", freq="D")
        equity_curve = pd.Series([100000 + i * 100 for i in range(len(dates))], index=dates)
        trades = pd.DataFrame({"pnl": [100, -50, 200]})

        metrics = calculate_metrics(equity_curve, trades)

        assert isinstance(metrics, PerformanceMetrics)
