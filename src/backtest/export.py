"""Export backtest results to various formats."""

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .metrics import PerformanceMetrics, calculate_monthly_returns


class BacktestExporter:
    """Export backtest results to JSON and other formats."""

    def __init__(self, output_dir: str = "data/results"):
        """Initialize the exporter.

        Args:
            output_dir: Directory for output files.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _serialize_value(self, value: Any) -> Any:
        """Serialize a value for JSON export.

        Args:
            value: Value to serialize.

        Returns:
            JSON-serializable value.
        """
        if isinstance(value, (date, datetime)):
            return value.isoformat()
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        if pd.isna(value):
            return None
        if isinstance(value, (float, int)):
            if pd.isna(value) or value != value:  # NaN check
                return None
            return value
        return value

    def _df_to_records(self, df: pd.DataFrame) -> list[dict]:
        """Convert DataFrame to list of records with serialization.

        Args:
            df: DataFrame to convert.

        Returns:
            List of dictionaries.
        """
        records = df.to_dict(orient="records")
        return [
            {k: self._serialize_value(v) for k, v in record.items()}
            for record in records
        ]

    def prepare_equity_curve(self, equity_curve: pd.Series) -> list[dict]:
        """Prepare equity curve for export.

        Args:
            equity_curve: Series of portfolio values.

        Returns:
            List of {date, value} records.
        """
        return [
            {"date": self._serialize_value(date_val), "value": float(value)}
            for date_val, value in equity_curve.items()
        ]

    def prepare_trades(self, trades: pd.DataFrame) -> list[dict]:
        """Prepare trades for export.

        Args:
            trades: DataFrame of trades.

        Returns:
            List of trade records.
        """
        return self._df_to_records(trades)

    def prepare_monthly_returns(self, equity_curve: pd.Series) -> list[dict]:
        """Prepare monthly returns for export.

        Args:
            equity_curve: Series of portfolio values.

        Returns:
            List of monthly return records.
        """
        monthly = calculate_monthly_returns(equity_curve)
        return self._df_to_records(monthly)

    def prepare_monthly_heatmap(self, equity_curve: pd.Series) -> dict[str, dict[str, float]]:
        """Prepare monthly returns as heatmap data.

        Args:
            equity_curve: Series of portfolio values.

        Returns:
            Nested dict: {year: {month: return}}.
        """
        monthly = calculate_monthly_returns(equity_curve)
        heatmap: dict[str, dict[str, float]] = {}

        for _, row in monthly.iterrows():
            year = str(int(row["year"]))
            month = str(int(row["month"]))

            if year not in heatmap:
                heatmap[year] = {}

            heatmap[year][month] = float(row["return"])

        return heatmap

    def export_json(
        self,
        metrics: PerformanceMetrics,
        equity_curve: pd.Series,
        trades: pd.DataFrame,
        config: dict,
        strategy_name: str,
        filename: str | None = None,
    ) -> Path:
        """Export complete backtest results to JSON.

        Args:
            metrics: Performance metrics.
            equity_curve: Portfolio value series.
            trades: Trade DataFrame.
            config: Configuration used.
            strategy_name: Name of the strategy.
            filename: Output filename (auto-generated if None).

        Returns:
            Path to the output file.
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"backtest_{strategy_name}_{timestamp}.json"

        # Ensure equity curve has proper datetime index
        equity_curve = equity_curve.copy()
        equity_curve.index = pd.to_datetime(equity_curve.index)

        result = {
            "metadata": {
                "strategy": strategy_name,
                "date_range": [
                    self._serialize_value(equity_curve.index[0]),
                    self._serialize_value(equity_curve.index[-1]),
                ],
                "generated_at": datetime.now().isoformat(),
                "config": config,
            },
            "summary": metrics.to_dict(),
            "equity_curve": self.prepare_equity_curve(equity_curve),
            "trades": self.prepare_trades(trades),
            "monthly_returns": self.prepare_monthly_returns(equity_curve),
            "monthly_heatmap": self.prepare_monthly_heatmap(equity_curve),
        }

        output_path = self.output_dir / filename

        with open(output_path, "w") as f:
            json.dump(result, f, indent=2, default=str)

        return output_path

    def export_csv(
        self,
        trades: pd.DataFrame,
        filename: str | None = None,
    ) -> Path:
        """Export trades to CSV.

        Args:
            trades: Trade DataFrame.
            filename: Output filename.

        Returns:
            Path to the output file.
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trades_{timestamp}.csv"

        output_path = self.output_dir / filename
        trades.to_csv(output_path, index=False)

        return output_path

    def export_for_dashboard(
        self,
        metrics: PerformanceMetrics,
        equity_curve: pd.Series,
        trades: pd.DataFrame,
        config: dict,
        strategy_name: str,
        dashboard_dir: str = "dashboard/public/data",
    ) -> Path:
        """Export results to dashboard data directory.

        Args:
            metrics: Performance metrics.
            equity_curve: Portfolio value series.
            trades: Trade DataFrame.
            config: Configuration used.
            strategy_name: Strategy name.
            dashboard_dir: Dashboard data directory.

        Returns:
            Path to the output file.
        """
        dashboard_path = Path(dashboard_dir)
        dashboard_path.mkdir(parents=True, exist_ok=True)

        # Create a standard filename for the dashboard
        filename = f"{strategy_name}_results.json"

        # Temporarily change output dir
        original_dir = self.output_dir
        self.output_dir = dashboard_path

        output_path = self.export_json(
            metrics, equity_curve, trades, config, strategy_name, filename
        )

        self.output_dir = original_dir

        return output_path

    def load_results(self, filepath: str | Path) -> dict:
        """Load results from JSON file.

        Args:
            filepath: Path to JSON file.

        Returns:
            Loaded results dictionary.
        """
        with open(filepath) as f:
            return json.load(f)

    def combine_results(
        self,
        result_files: list[Path],
        output_filename: str = "combined_results.json",
    ) -> Path:
        """Combine multiple backtest results for comparison.

        Args:
            result_files: List of result JSON files.
            output_filename: Output filename.

        Returns:
            Path to combined results file.
        """
        combined = {
            "strategies": {},
            "generated_at": datetime.now().isoformat(),
        }

        for filepath in result_files:
            results = self.load_results(filepath)
            strategy_name = results["metadata"]["strategy"]
            combined["strategies"][strategy_name] = results

        output_path = self.output_dir / output_filename

        with open(output_path, "w") as f:
            json.dump(combined, f, indent=2)

        return output_path
