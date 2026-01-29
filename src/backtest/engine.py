"""Core backtesting engine."""

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import pandas as pd

from ..strategies.base_strategy import BaseStrategy, Signal, SignalType, Position
from .metrics import PerformanceMetrics, MetricsCalculator
from .export import BacktestExporter

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Completed trade record."""

    ticker: str
    direction: str
    entry_date: date
    exit_date: date
    entry_price: float
    exit_price: float
    shares: float
    pnl: float
    return_pct: float
    entry_sentiment: float | None = None
    exit_reason: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "ticker": self.ticker,
            "direction": self.direction,
            "entry_date": self.entry_date.isoformat() if isinstance(self.entry_date, date) else str(self.entry_date),
            "exit_date": self.exit_date.isoformat() if isinstance(self.exit_date, date) else str(self.exit_date),
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "shares": self.shares,
            "pnl": self.pnl,
            "return_pct": self.return_pct,
            "entry_sentiment": self.entry_sentiment,
            "exit_reason": self.exit_reason,
            "metadata": self.metadata,
        }


@dataclass
class BacktestResult:
    """Complete backtest results."""

    strategy_name: str
    start_date: date
    end_date: date
    initial_capital: float
    final_capital: float
    equity_curve: pd.Series
    trades: list[Trade]
    metrics: PerformanceMetrics
    config: dict

    def to_dataframe(self) -> pd.DataFrame:
        """Convert trades to DataFrame."""
        return pd.DataFrame([t.to_dict() for t in self.trades])


class BacktestEngine:
    """Vectorized backtesting engine."""

    def __init__(
        self,
        initial_capital: float = 100000,
        transaction_cost: float = 0.001,
        slippage: float = 0.0005,
    ):
        """Initialize the backtest engine.

        Args:
            initial_capital: Starting capital.
            transaction_cost: Transaction cost as fraction.
            slippage: Slippage as fraction.
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage

        self.metrics_calculator = MetricsCalculator()
        self.exporter = BacktestExporter()

    def run(
        self,
        strategy: BaseStrategy,
        sentiment_data: pd.DataFrame,
        price_data: dict[str, pd.DataFrame],
        start_date: date | str,
        end_date: date | str,
        benchmark_ticker: str | None = "SPY",
    ) -> BacktestResult:
        """Run a backtest.

        Args:
            strategy: Strategy instance.
            sentiment_data: DataFrame with sentiment data.
            price_data: Dict of ticker -> price DataFrame.
            start_date: Backtest start date.
            end_date: Backtest end date.
            benchmark_ticker: Benchmark ticker for comparison.

        Returns:
            BacktestResult with all metrics and trades.
        """
        if isinstance(start_date, str):
            start_date = pd.Timestamp(start_date).date()
        if isinstance(end_date, str):
            end_date = pd.Timestamp(end_date).date()

        # Initialize state
        capital = self.initial_capital
        positions: list[Position] = []
        trades: list[Trade] = []
        equity_curve: dict[date, float] = {}

        # Get all trading days
        trading_days = self._get_trading_days(price_data, start_date, end_date)

        logger.info(
            f"Running backtest for {strategy.name} from {start_date} to {end_date}"
        )

        for current_date in trading_days:
            # Calculate current portfolio value
            portfolio_value = capital + sum(
                self._get_position_value(p, price_data, current_date)
                for p in positions
            )
            equity_curve[current_date] = portfolio_value

            # Generate signals
            signals = strategy.generate_signals(
                sentiment_data, price_data, current_date, positions
            )

            # Process signals
            for signal in signals:
                if signal.signal_type == SignalType.CLOSE:
                    # Close position
                    position = self._find_position(positions, signal.ticker)
                    if position:
                        trade = self._close_position(
                            position, price_data, current_date, signal.reason
                        )
                        if trade:
                            trades.append(trade)
                            capital += trade.pnl + (position.entry_price * position.shares)
                            positions.remove(position)

                elif signal.signal_type in (SignalType.LONG, SignalType.SHORT):
                    # Check if we can open new position
                    if len(positions) >= strategy.max_positions:
                        continue

                    # Open position
                    position = self._open_position(
                        signal, price_data, current_date, capital, strategy
                    )
                    if position:
                        capital -= position.entry_price * position.shares
                        positions.append(position)

        # Close remaining positions at end
        final_date = trading_days[-1] if trading_days else end_date
        for position in positions[:]:
            trade = self._close_position(
                position, price_data, final_date, "end_of_backtest"
            )
            if trade:
                trades.append(trade)
                capital += trade.pnl + (position.entry_price * position.shares)

        # Calculate final equity
        final_capital = capital
        equity_curve[final_date] = final_capital

        # Convert equity curve to series
        equity_series = pd.Series(equity_curve).sort_index()

        # Calculate trades DataFrame
        trades_df = pd.DataFrame([t.to_dict() for t in trades])

        # Get benchmark returns if available
        benchmark_returns = None
        if benchmark_ticker and benchmark_ticker in price_data:
            benchmark_returns = self._calculate_returns(
                price_data[benchmark_ticker], start_date, end_date
            )

        # Calculate metrics
        metrics = self.metrics_calculator.calculate_all_metrics(
            equity_series, trades_df, benchmark_returns
        )

        return BacktestResult(
            strategy_name=strategy.name,
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
            final_capital=final_capital,
            equity_curve=equity_series,
            trades=trades,
            metrics=metrics,
            config=strategy.get_config(),
        )

    def _get_trading_days(
        self,
        price_data: dict[str, pd.DataFrame],
        start_date: date,
        end_date: date,
    ) -> list[date]:
        """Get list of trading days from price data.

        Args:
            price_data: Dict of price DataFrames.
            start_date: Start date.
            end_date: End date.

        Returns:
            Sorted list of trading days.
        """
        all_dates = set()

        for df in price_data.values():
            if df.empty:
                continue
            dates = pd.to_datetime(df["date"]).dt.date
            valid_dates = dates[(dates >= start_date) & (dates <= end_date)]
            all_dates.update(valid_dates)

        return sorted(all_dates)

    def _get_position_value(
        self,
        position: Position,
        price_data: dict[str, pd.DataFrame],
        current_date: date,
    ) -> float:
        """Get current value of a position.

        Args:
            position: Position to value.
            price_data: Dict of price DataFrames.
            current_date: Current date.

        Returns:
            Current position value.
        """
        current_price = self._get_price(price_data, position.ticker, current_date)
        if current_price is None:
            return position.entry_price * position.shares

        return position.calculate_pnl(current_price)

    def _get_price(
        self,
        price_data: dict[str, pd.DataFrame],
        ticker: str,
        target_date: date,
        price_col: str = "adjusted_close",
    ) -> float | None:
        """Get price for a ticker on a date.

        Args:
            price_data: Dict of price DataFrames.
            ticker: Ticker symbol.
            target_date: Date to get price for.
            price_col: Price column to use.

        Returns:
            Price or None.
        """
        if ticker not in price_data:
            return None

        df = price_data[ticker]
        df = df[pd.to_datetime(df["date"]).dt.date <= target_date]

        if df.empty:
            return None

        return float(df.iloc[-1][price_col])

    def _find_position(
        self,
        positions: list[Position],
        ticker: str,
    ) -> Position | None:
        """Find a position by ticker.

        Args:
            positions: List of positions.
            ticker: Ticker to find.

        Returns:
            Position or None.
        """
        for p in positions:
            if p.ticker == ticker:
                return p
        return None

    def _open_position(
        self,
        signal: Signal,
        price_data: dict[str, pd.DataFrame],
        current_date: date,
        available_capital: float,
        strategy: BaseStrategy,
    ) -> Position | None:
        """Open a new position.

        Args:
            signal: Entry signal.
            price_data: Dict of price DataFrames.
            current_date: Current date.
            available_capital: Available capital.
            strategy: Strategy instance.

        Returns:
            New Position or None.
        """
        price = self._get_price(price_data, signal.ticker, current_date)
        if price is None or price <= 0:
            return None

        # Apply slippage
        if signal.signal_type == SignalType.LONG:
            entry_price = price * (1 + self.slippage)
            direction = "long"
        else:
            entry_price = price * (1 - self.slippage)
            direction = "short"

        # Calculate position size
        shares = strategy.calculate_position_size(
            available_capital, entry_price, signal.strength
        )

        if shares <= 0:
            return None

        return Position(
            ticker=signal.ticker,
            entry_date=current_date,
            entry_price=entry_price,
            shares=shares,
            direction=direction,
            entry_sentiment=signal.sentiment,
            stop_loss=strategy.stop_loss,
            take_profit=strategy.take_profit,
            metadata=signal.metadata,
        )

    def _close_position(
        self,
        position: Position,
        price_data: dict[str, pd.DataFrame],
        current_date: date,
        reason: str,
    ) -> Trade | None:
        """Close a position and create trade record.

        Args:
            position: Position to close.
            price_data: Dict of price DataFrames.
            current_date: Current date.
            reason: Exit reason.

        Returns:
            Trade record or None.
        """
        price = self._get_price(price_data, position.ticker, current_date)
        if price is None:
            price = position.entry_price  # Use entry price if no current price

        # Apply slippage
        if position.is_long:
            exit_price = price * (1 - self.slippage)
        else:
            exit_price = price * (1 + self.slippage)

        # Calculate PnL
        pnl = position.calculate_pnl(exit_price)

        # Apply transaction cost
        pnl -= (position.entry_price + exit_price) * position.shares * self.transaction_cost

        return_pct = position.calculate_return(exit_price)

        return Trade(
            ticker=position.ticker,
            direction=position.direction,
            entry_date=position.entry_date,
            exit_date=current_date,
            entry_price=position.entry_price,
            exit_price=exit_price,
            shares=position.shares,
            pnl=pnl,
            return_pct=return_pct,
            entry_sentiment=position.entry_sentiment,
            exit_reason=reason,
            metadata=position.metadata,
        )

    def _calculate_returns(
        self,
        price_df: pd.DataFrame,
        start_date: date,
        end_date: date,
    ) -> pd.Series:
        """Calculate daily returns for a price series.

        Args:
            price_df: Price DataFrame.
            start_date: Start date.
            end_date: End date.

        Returns:
            Series of daily returns.
        """
        df = price_df.copy()
        df["date"] = pd.to_datetime(df["date"]).dt.date

        df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
        df = df.set_index("date").sort_index()

        returns = df["adjusted_close"].pct_change().dropna()
        returns.index = pd.to_datetime(returns.index)

        return returns

    def export_results(
        self,
        result: BacktestResult,
        output_dir: str = "data/results",
    ) -> str:
        """Export backtest results to JSON.

        Args:
            result: BacktestResult to export.
            output_dir: Output directory.

        Returns:
            Path to exported file.
        """
        self.exporter.output_dir = output_dir
        trades_df = pd.DataFrame([t.to_dict() for t in result.trades])

        path = self.exporter.export_json(
            result.metrics,
            result.equity_curve,
            trades_df,
            result.config,
            result.strategy_name,
        )

        return str(path)


def run_backtest(
    strategy: BaseStrategy,
    sentiment_data: pd.DataFrame,
    price_data: dict[str, pd.DataFrame],
    start_date: date | str,
    end_date: date | str,
    initial_capital: float = 100000,
    benchmark_ticker: str | None = "SPY",
) -> BacktestResult:
    """Convenience function to run a backtest.

    Args:
        strategy: Strategy instance.
        sentiment_data: Sentiment DataFrame.
        price_data: Price data dict.
        start_date: Start date.
        end_date: End date.
        initial_capital: Starting capital.
        benchmark_ticker: Benchmark ticker.

    Returns:
        BacktestResult.
    """
    engine = BacktestEngine(initial_capital=initial_capital)
    return engine.run(
        strategy, sentiment_data, price_data, start_date, end_date, benchmark_ticker
    )
