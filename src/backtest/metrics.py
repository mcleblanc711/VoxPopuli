"""Performance metrics calculation for backtests."""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import date


@dataclass
class PerformanceMetrics:
    """Container for backtest performance metrics."""

    total_return: float
    cagr: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration: int  # days
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_holding_period: float  # days
    exposure_time: float  # percentage of time in market
    volatility: float  # annualized
    calmar_ratio: float  # CAGR / max drawdown
    benchmark_return: float | None = None
    alpha: float | None = None
    beta: float | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "total_return": self.total_return,
            "cagr": self.cagr,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_duration": self.max_drawdown_duration,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "avg_holding_period": self.avg_holding_period,
            "exposure_time": self.exposure_time,
            "volatility": self.volatility,
            "calmar_ratio": self.calmar_ratio,
            "benchmark_return": self.benchmark_return,
            "alpha": self.alpha,
            "beta": self.beta,
        }


class MetricsCalculator:
    """Calculate performance metrics from backtest results."""

    TRADING_DAYS_PER_YEAR = 252
    RISK_FREE_RATE = 0.02  # 2% annual

    def __init__(self, risk_free_rate: float = 0.02):
        """Initialize the metrics calculator.

        Args:
            risk_free_rate: Annual risk-free rate.
        """
        self.risk_free_rate = risk_free_rate

    def calculate_returns(self, equity_curve: pd.Series) -> pd.Series:
        """Calculate daily returns from equity curve.

        Args:
            equity_curve: Series of portfolio values indexed by date.

        Returns:
            Series of daily returns.
        """
        return equity_curve.pct_change().dropna()

    def calculate_total_return(self, equity_curve: pd.Series) -> float:
        """Calculate total return.

        Args:
            equity_curve: Series of portfolio values.

        Returns:
            Total return as decimal (e.g., 0.5 for 50%).
        """
        return (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1

    def calculate_cagr(
        self,
        equity_curve: pd.Series,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> float:
        """Calculate Compound Annual Growth Rate.

        Args:
            equity_curve: Series of portfolio values.
            start_date: Start date (uses index if None).
            end_date: End date (uses index if None).

        Returns:
            CAGR as decimal.
        """
        if start_date is None:
            start_date = equity_curve.index[0]
        if end_date is None:
            end_date = equity_curve.index[-1]

        years = (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days / 365.25
        if years <= 0:
            return 0.0

        total_return = self.calculate_total_return(equity_curve)
        return (1 + total_return) ** (1 / years) - 1

    def calculate_volatility(
        self,
        returns: pd.Series,
        annualize: bool = True,
    ) -> float:
        """Calculate volatility (standard deviation of returns).

        Args:
            returns: Series of returns.
            annualize: Whether to annualize.

        Returns:
            Volatility.
        """
        vol = returns.std()
        if annualize:
            vol *= np.sqrt(self.TRADING_DAYS_PER_YEAR)
        return vol

    def calculate_sharpe_ratio(
        self,
        returns: pd.Series,
        risk_free_rate: float | None = None,
    ) -> float:
        """Calculate Sharpe ratio.

        Args:
            returns: Series of daily returns.
            risk_free_rate: Annual risk-free rate (uses default if None).

        Returns:
            Sharpe ratio.
        """
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate

        excess_returns = returns - (risk_free_rate / self.TRADING_DAYS_PER_YEAR)
        if returns.std() == 0:
            return 0.0
        return (excess_returns.mean() / returns.std()) * np.sqrt(self.TRADING_DAYS_PER_YEAR)

    def calculate_sortino_ratio(
        self,
        returns: pd.Series,
        risk_free_rate: float | None = None,
    ) -> float:
        """Calculate Sortino ratio (uses downside deviation).

        Args:
            returns: Series of daily returns.
            risk_free_rate: Annual risk-free rate.

        Returns:
            Sortino ratio.
        """
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate

        excess_returns = returns - (risk_free_rate / self.TRADING_DAYS_PER_YEAR)
        downside_returns = returns[returns < 0]

        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0

        return (excess_returns.mean() / downside_returns.std()) * np.sqrt(
            self.TRADING_DAYS_PER_YEAR
        )

    def calculate_max_drawdown(
        self,
        equity_curve: pd.Series,
    ) -> tuple[float, int]:
        """Calculate maximum drawdown and duration.

        Args:
            equity_curve: Series of portfolio values.

        Returns:
            Tuple of (max_drawdown, duration_in_days).
        """
        rolling_max = equity_curve.expanding().max()
        drawdowns = (equity_curve - rolling_max) / rolling_max

        max_drawdown = drawdowns.min()

        # Calculate duration
        in_drawdown = drawdowns < 0
        if not in_drawdown.any():
            return 0.0, 0

        # Find consecutive drawdown periods
        drawdown_groups = (in_drawdown != in_drawdown.shift()).cumsum()
        drawdown_groups = drawdown_groups[in_drawdown]

        if len(drawdown_groups) == 0:
            return float(max_drawdown), 0

        max_duration = drawdown_groups.groupby(drawdown_groups).size().max()

        return float(max_drawdown), int(max_duration)

    def calculate_win_rate(self, trades: pd.DataFrame) -> float:
        """Calculate win rate from trades.

        Args:
            trades: DataFrame with 'pnl' column.

        Returns:
            Win rate as decimal.
        """
        if len(trades) == 0:
            return 0.0
        return (trades["pnl"] > 0).sum() / len(trades)

    def calculate_profit_factor(self, trades: pd.DataFrame) -> float:
        """Calculate profit factor (gross profit / gross loss).

        Args:
            trades: DataFrame with 'pnl' column.

        Returns:
            Profit factor.
        """
        gross_profit = trades[trades["pnl"] > 0]["pnl"].sum()
        gross_loss = abs(trades[trades["pnl"] < 0]["pnl"].sum())

        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0

        return gross_profit / gross_loss

    def calculate_trade_stats(
        self,
        trades: pd.DataFrame,
    ) -> dict:
        """Calculate trade statistics.

        Args:
            trades: DataFrame with trade data.

        Returns:
            Dict with trade statistics.
        """
        if len(trades) == 0:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "avg_holding_period": 0.0,
            }

        winning = trades[trades["pnl"] > 0]
        losing = trades[trades["pnl"] < 0]

        avg_holding = 0.0
        if "entry_date" in trades.columns and "exit_date" in trades.columns:
            holding_periods = (
                pd.to_datetime(trades["exit_date"]) - pd.to_datetime(trades["entry_date"])
            ).dt.days
            avg_holding = holding_periods.mean()

        return {
            "total_trades": len(trades),
            "winning_trades": len(winning),
            "losing_trades": len(losing),
            "avg_win": winning["pnl"].mean() if len(winning) > 0 else 0.0,
            "avg_loss": losing["pnl"].mean() if len(losing) > 0 else 0.0,
            "avg_holding_period": avg_holding,
        }

    def calculate_exposure_time(
        self,
        equity_curve: pd.Series,
        positions: pd.Series | None = None,
    ) -> float:
        """Calculate percentage of time with open positions.

        Args:
            equity_curve: Series of portfolio values.
            positions: Series indicating position status (optional).

        Returns:
            Exposure time as decimal.
        """
        if positions is None:
            # Assume exposed when equity changes
            returns = equity_curve.pct_change()
            exposed = returns != 0
            return exposed.sum() / len(equity_curve)

        return (positions != 0).sum() / len(positions)

    def calculate_benchmark_comparison(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series,
    ) -> dict:
        """Calculate alpha and beta vs benchmark.

        Args:
            returns: Strategy returns.
            benchmark_returns: Benchmark returns.

        Returns:
            Dict with benchmark metrics.
        """
        # Align returns
        common_dates = returns.index.intersection(benchmark_returns.index)
        if len(common_dates) == 0:
            return {"benchmark_return": None, "alpha": None, "beta": None}

        strat_returns = returns.loc[common_dates]
        bench_returns = benchmark_returns.loc[common_dates]

        # Calculate beta
        covariance = strat_returns.cov(bench_returns)
        variance = bench_returns.var()
        beta = covariance / variance if variance != 0 else 0

        # Calculate alpha (annualized)
        strat_annual = (1 + strat_returns.mean()) ** self.TRADING_DAYS_PER_YEAR - 1
        bench_annual = (1 + bench_returns.mean()) ** self.TRADING_DAYS_PER_YEAR - 1
        alpha = strat_annual - (self.risk_free_rate + beta * (bench_annual - self.risk_free_rate))

        benchmark_total = (1 + bench_returns).prod() - 1

        return {
            "benchmark_return": benchmark_total,
            "alpha": alpha,
            "beta": beta,
        }

    def calculate_all_metrics(
        self,
        equity_curve: pd.Series,
        trades: pd.DataFrame,
        benchmark_returns: pd.Series | None = None,
    ) -> PerformanceMetrics:
        """Calculate all performance metrics.

        Args:
            equity_curve: Series of portfolio values indexed by date.
            trades: DataFrame of trades with entry/exit dates, prices, pnl.
            benchmark_returns: Optional benchmark returns for comparison.

        Returns:
            PerformanceMetrics object.
        """
        returns = self.calculate_returns(equity_curve)
        total_return = self.calculate_total_return(equity_curve)
        cagr = self.calculate_cagr(equity_curve)
        volatility = self.calculate_volatility(returns)
        sharpe = self.calculate_sharpe_ratio(returns)
        sortino = self.calculate_sortino_ratio(returns)
        max_dd, max_dd_duration = self.calculate_max_drawdown(equity_curve)
        calmar = abs(cagr / max_dd) if max_dd != 0 else 0.0

        win_rate = self.calculate_win_rate(trades)
        profit_factor = self.calculate_profit_factor(trades)
        trade_stats = self.calculate_trade_stats(trades)
        exposure = self.calculate_exposure_time(equity_curve)

        benchmark_metrics = {"benchmark_return": None, "alpha": None, "beta": None}
        if benchmark_returns is not None:
            benchmark_metrics = self.calculate_benchmark_comparison(returns, benchmark_returns)

        return PerformanceMetrics(
            total_return=total_return,
            cagr=cagr,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            max_drawdown_duration=max_dd_duration,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=trade_stats["avg_win"],
            avg_loss=trade_stats["avg_loss"],
            total_trades=trade_stats["total_trades"],
            winning_trades=trade_stats["winning_trades"],
            losing_trades=trade_stats["losing_trades"],
            avg_holding_period=trade_stats["avg_holding_period"],
            exposure_time=exposure,
            volatility=volatility,
            calmar_ratio=calmar,
            benchmark_return=benchmark_metrics["benchmark_return"],
            alpha=benchmark_metrics["alpha"],
            beta=benchmark_metrics["beta"],
        )


def calculate_metrics(
    equity_curve: pd.Series,
    trades: pd.DataFrame,
    benchmark_returns: pd.Series | None = None,
    risk_free_rate: float = 0.02,
) -> PerformanceMetrics:
    """Convenience function to calculate all metrics.

    Args:
        equity_curve: Series of portfolio values.
        trades: DataFrame of trades.
        benchmark_returns: Optional benchmark returns.
        risk_free_rate: Annual risk-free rate.

    Returns:
        PerformanceMetrics object.
    """
    calculator = MetricsCalculator(risk_free_rate=risk_free_rate)
    return calculator.calculate_all_metrics(equity_curve, trades, benchmark_returns)


def calculate_monthly_returns(equity_curve: pd.Series) -> pd.DataFrame:
    """Calculate monthly returns from equity curve.

    Args:
        equity_curve: Series of portfolio values indexed by date.

    Returns:
        DataFrame with year, month, and return columns.
    """
    equity_curve.index = pd.to_datetime(equity_curve.index)

    # Resample to month-end
    monthly = equity_curve.resample("ME").last()
    monthly_returns = monthly.pct_change().dropna()

    result = pd.DataFrame({
        "year": monthly_returns.index.year,
        "month": monthly_returns.index.month,
        "return": monthly_returns.values,
    })

    return result


def format_metrics_table(metrics: PerformanceMetrics) -> str:
    """Format metrics as a readable table.

    Args:
        metrics: PerformanceMetrics object.

    Returns:
        Formatted string table.
    """
    lines = [
        "=" * 50,
        "PERFORMANCE METRICS",
        "=" * 50,
        f"Total Return:       {metrics.total_return:>10.2%}",
        f"CAGR:               {metrics.cagr:>10.2%}",
        f"Sharpe Ratio:       {metrics.sharpe_ratio:>10.2f}",
        f"Sortino Ratio:      {metrics.sortino_ratio:>10.2f}",
        f"Max Drawdown:       {metrics.max_drawdown:>10.2%}",
        f"Calmar Ratio:       {metrics.calmar_ratio:>10.2f}",
        "-" * 50,
        f"Total Trades:       {metrics.total_trades:>10}",
        f"Win Rate:           {metrics.win_rate:>10.2%}",
        f"Profit Factor:      {metrics.profit_factor:>10.2f}",
        f"Avg Win:            {metrics.avg_win:>10.2f}",
        f"Avg Loss:           {metrics.avg_loss:>10.2f}",
        f"Avg Holding Period: {metrics.avg_holding_period:>10.1f} days",
        "-" * 50,
        f"Volatility (Ann.):  {metrics.volatility:>10.2%}",
        f"Exposure Time:      {metrics.exposure_time:>10.2%}",
    ]

    if metrics.benchmark_return is not None:
        lines.extend([
            "-" * 50,
            f"Benchmark Return:   {metrics.benchmark_return:>10.2%}",
            f"Alpha:              {metrics.alpha:>10.2%}",
            f"Beta:               {metrics.beta:>10.2f}",
        ])

    lines.append("=" * 50)
    return "\n".join(lines)
