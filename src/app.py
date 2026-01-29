#!/usr/bin/env python3
"""VoxPopuli CLI Application."""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add parent directory to path for direct script execution
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

import typer
import yaml
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.data.reddit_ingester import RedditIngester
from src.data.price_fetcher import PriceFetcher
from src.data.ticker_extractor import TickerExtractor
from src.data.sentiment_scorer import SentimentScorer
from src.strategies import get_strategy, STRATEGY_REGISTRY
from src.backtest.engine import BacktestEngine
from src.backtest.metrics import format_metrics_table

app = typer.Typer(
    name="voxpopuli",
    help="Reddit sentiment-based trading backtester",
    add_completion=False,
)
console = Console()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("voxpopuli")


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    path = Path(config_path)
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f)
    return {}


@app.command()
def fetch_reddit(
    subreddits: str = typer.Option(
        "wallstreetbets,stocks",
        "--subreddits", "-s",
        help="Comma-separated list of subreddits",
    ),
    start: str = typer.Option(
        ...,
        "--start",
        help="Start month (YYYY-MM format)",
    ),
    end: str = typer.Option(
        ...,
        "--end",
        help="End month (YYYY-MM format)",
    ),
    data_type: str = typer.Option(
        "submissions",
        "--type", "-t",
        help="Data type: submissions or comments",
    ),
    min_score: int = typer.Option(
        1,
        "--min-score",
        help="Minimum score threshold",
    ),
    output_dir: str = typer.Option(
        "data/reddit",
        "--output", "-o",
        help="Output directory for parquet files",
    ),
):
    """Fetch Reddit data from Arctic Shift dumps."""
    subreddit_list = [s.strip() for s in subreddits.split(",")]

    # Parse dates
    start_year, start_month = map(int, start.split("-"))
    end_year, end_month = map(int, end.split("-"))

    console.print(f"[bold]Fetching Reddit {data_type}[/bold]")
    console.print(f"  Subreddits: {', '.join(subreddit_list)}")
    console.print(f"  Range: {start} to {end}")

    ingester = RedditIngester(data_dir=output_dir)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Downloading and processing...", total=None)

        files = ingester.ingest_range(
            subreddits=subreddit_list,
            start_year=start_year,
            start_month=start_month,
            end_year=end_year,
            end_month=end_month,
            data_type=data_type,
            min_score=min_score,
        )

        progress.update(task, completed=True)

    console.print(f"\n[green]Created {len(files)} parquet files[/green]")
    for f in files:
        console.print(f"  - {f}")


@app.command()
def fetch_prices(
    tickers: str = typer.Option(
        ...,
        "--tickers", "-t",
        help="Comma-separated list of tickers",
    ),
    start: str = typer.Option(
        ...,
        "--start",
        help="Start date (YYYY-MM-DD)",
    ),
    end: str = typer.Option(
        ...,
        "--end",
        help="End date (YYYY-MM-DD)",
    ),
    source: str = typer.Option(
        "yfinance",
        "--source", "-s",
        help="Data source: ibkr, yfinance, or auto",
    ),
    output_dir: str = typer.Option(
        "data/prices",
        "--output", "-o",
        help="Output directory for parquet files",
    ),
):
    """Fetch price data from IBKR or yfinance."""
    ticker_list = [t.strip().upper() for t in tickers.split(",")]

    console.print(f"[bold]Fetching price data[/bold]")
    console.print(f"  Tickers: {', '.join(ticker_list)}")
    console.print(f"  Range: {start} to {end}")
    console.print(f"  Source: {source}")

    fetcher = PriceFetcher(data_dir=output_dir)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Fetching prices...", total=len(ticker_list))

        results = {}
        for ticker in ticker_list:
            progress.update(task, description=f"Fetching {ticker}...")
            df = fetcher.cache_prices(ticker, start, end, source)
            if df is not None:
                results[ticker] = len(df)
            progress.advance(task)

    console.print(f"\n[green]Fetched data for {len(results)} tickers[/green]")

    table = Table(title="Price Data Summary")
    table.add_column("Ticker")
    table.add_column("Rows", justify="right")

    for ticker, rows in results.items():
        table.add_row(ticker, str(rows))

    console.print(table)


@app.command()
def run_backtest(
    strategy: str = typer.Option(
        "attention_momentum",
        "--strategy", "-s",
        help=f"Strategy name: {', '.join(STRATEGY_REGISTRY.keys())}",
    ),
    config_file: str = typer.Option(
        "config.yaml",
        "--config", "-c",
        help="Path to config file",
    ),
    reddit_dir: str = typer.Option(
        "data/reddit",
        "--reddit-dir",
        help="Reddit data directory",
    ),
    price_dir: str = typer.Option(
        "data/prices",
        "--price-dir",
        help="Price data directory",
    ),
    output_dir: str = typer.Option(
        "data/results",
        "--output", "-o",
        help="Output directory for results",
    ),
    start: Optional[str] = typer.Option(
        None,
        "--start",
        help="Start date (YYYY-MM-DD), overrides config",
    ),
    end: Optional[str] = typer.Option(
        None,
        "--end",
        help="End date (YYYY-MM-DD), overrides config",
    ),
    capital: float = typer.Option(
        100000,
        "--capital",
        help="Initial capital",
    ),
):
    """Run a backtest with the specified strategy."""
    import pandas as pd
    from src.data.ticker_extractor import TickerExtractor
    from src.data.sentiment_scorer import SentimentScorer

    # Load config
    config = load_config(config_file)

    # Get date range
    date_config = config.get("data", {}).get("date_range", {})
    start_date = start or date_config.get("start", "2022-01-01")
    end_date = end or date_config.get("end", "2023-12-31")

    console.print(f"[bold]Running backtest[/bold]")
    console.print(f"  Strategy: {strategy}")
    console.print(f"  Date range: {start_date} to {end_date}")
    console.print(f"  Initial capital: ${capital:,.2f}")

    # Load Reddit data
    console.print("\n[dim]Loading Reddit data...[/dim]")
    reddit_path = Path(reddit_dir)
    reddit_files = list(reddit_path.glob("*.parquet"))

    if not reddit_files:
        console.print("[red]No Reddit data found. Run fetch-reddit first.[/red]")
        raise typer.Exit(1)

    sentiment_data = pd.concat([pd.read_parquet(f) for f in reddit_files])
    console.print(f"  Loaded {len(sentiment_data)} posts")

    # Extract tickers
    console.print("[dim]Extracting tickers...[/dim]")
    extractor = TickerExtractor()
    sentiment_data = extractor.add_ticker_column(sentiment_data)

    # Score sentiment
    console.print("[dim]Scoring sentiment...[/dim]")
    scorer = SentimentScorer(model=config.get("sentiment", {}).get("model", "vader"))
    sentiment_data = scorer.score_dataframe(sentiment_data)

    # Load price data
    console.print("[dim]Loading price data...[/dim]")
    price_path = Path(price_dir)
    price_files = list(price_path.glob("*.parquet"))

    if not price_files:
        console.print("[red]No price data found. Run fetch-prices first.[/red]")
        raise typer.Exit(1)

    price_data = {}
    for f in price_files:
        ticker = f.stem.replace("_daily", "")
        price_data[ticker] = pd.read_parquet(f)

    console.print(f"  Loaded prices for {len(price_data)} tickers")

    # Initialize strategy
    strategy_params = config.get("strategy", {}).get("params", {})
    backtest_config = config.get("backtest", {})

    strategy_instance = get_strategy(
        strategy,
        position_size=backtest_config.get("position_size", 0.1),
        max_positions=backtest_config.get("max_positions", 10),
        allow_shorts=backtest_config.get("allow_shorts", False),
        transaction_cost=backtest_config.get("transaction_cost", 0.001),
        **strategy_params,
    )

    # Run backtest
    console.print("\n[dim]Running backtest...[/dim]")
    engine = BacktestEngine(
        initial_capital=capital,
        transaction_cost=backtest_config.get("transaction_cost", 0.001),
        slippage=backtest_config.get("slippage", 0.0005),
    )

    result = engine.run(
        strategy=strategy_instance,
        sentiment_data=sentiment_data,
        price_data=price_data,
        start_date=start_date,
        end_date=end_date,
        benchmark_ticker=backtest_config.get("benchmark", "SPY"),
    )

    # Display results
    console.print("\n" + format_metrics_table(result.metrics))

    # Export results
    output_path = engine.export_results(result, output_dir)
    console.print(f"\n[green]Results exported to: {output_path}[/green]")


@app.command()
def export_results(
    input_file: str = typer.Option(
        ...,
        "--input", "-i",
        help="Input pickle file from backtest",
    ),
    output_file: str = typer.Option(
        ...,
        "--output", "-o",
        help="Output JSON file",
    ),
    dashboard: bool = typer.Option(
        False,
        "--dashboard",
        help="Also export to dashboard directory",
    ),
):
    """Export backtest results to JSON format."""
    import pickle

    console.print(f"[bold]Exporting results[/bold]")
    console.print(f"  Input: {input_file}")
    console.print(f"  Output: {output_file}")

    # Load results
    with open(input_file, "rb") as f:
        result = pickle.load(f)

    # Export
    from src.backtest.export import BacktestExporter
    import pandas as pd

    exporter = BacktestExporter()
    trades_df = pd.DataFrame([t.to_dict() for t in result.trades])

    path = exporter.export_json(
        result.metrics,
        result.equity_curve,
        trades_df,
        result.config,
        result.strategy_name,
        output_file,
    )

    console.print(f"[green]Exported to: {path}[/green]")

    if dashboard:
        dashboard_path = exporter.export_for_dashboard(
            result.metrics,
            result.equity_curve,
            trades_df,
            result.config,
            result.strategy_name,
        )
        console.print(f"[green]Dashboard data: {dashboard_path}[/green]")


@app.command()
def list_strategies():
    """List available trading strategies."""
    table = Table(title="Available Strategies")
    table.add_column("Name")
    table.add_column("Description")

    for name, cls in STRATEGY_REGISTRY.items():
        table.add_row(name, cls.description)

    console.print(table)


@app.command()
def analyze_sentiment(
    input_file: str = typer.Option(
        ...,
        "--input", "-i",
        help="Input parquet file with Reddit data",
    ),
    model: str = typer.Option(
        "vader",
        "--model", "-m",
        help="Sentiment model: vader or finbert",
    ),
    output_file: Optional[str] = typer.Option(
        None,
        "--output", "-o",
        help="Output parquet file (optional)",
    ),
):
    """Analyze sentiment in Reddit data."""
    import pandas as pd

    console.print(f"[bold]Analyzing sentiment[/bold]")
    console.print(f"  Input: {input_file}")
    console.print(f"  Model: {model}")

    # Load data
    df = pd.read_parquet(input_file)
    console.print(f"  Loaded {len(df)} posts")

    # Score sentiment
    scorer = SentimentScorer(model=model)
    df = scorer.score_dataframe(df)

    # Summary stats
    console.print("\n[bold]Sentiment Summary[/bold]")
    console.print(f"  Mean: {df['sentiment'].mean():.3f}")
    console.print(f"  Std:  {df['sentiment'].std():.3f}")
    console.print(f"  Min:  {df['sentiment'].min():.3f}")
    console.print(f"  Max:  {df['sentiment'].max():.3f}")

    # Distribution
    positive = (df["sentiment"] > 0.1).sum()
    negative = (df["sentiment"] < -0.1).sum()
    neutral = len(df) - positive - negative

    console.print(f"\n  Positive: {positive} ({positive/len(df)*100:.1f}%)")
    console.print(f"  Neutral:  {neutral} ({neutral/len(df)*100:.1f}%)")
    console.print(f"  Negative: {negative} ({negative/len(df)*100:.1f}%)")

    # Save if output specified
    if output_file:
        df.to_parquet(output_file)
        console.print(f"\n[green]Saved to: {output_file}[/green]")


@app.command()
def extract_tickers(
    input_file: str = typer.Option(
        ...,
        "--input", "-i",
        help="Input parquet file with Reddit data",
    ),
    ticker_list: Optional[str] = typer.Option(
        None,
        "--ticker-list",
        help="Path to CSV with valid tickers",
    ),
    top_n: int = typer.Option(
        20,
        "--top", "-n",
        help="Show top N tickers",
    ),
):
    """Extract and count ticker mentions."""
    import pandas as pd

    console.print(f"[bold]Extracting tickers[/bold]")
    console.print(f"  Input: {input_file}")

    # Load data
    df = pd.read_parquet(input_file)
    console.print(f"  Loaded {len(df)} posts")

    # Initialize extractor
    extractor = TickerExtractor(ticker_list)

    # Count tickers
    counts = extractor.count_tickers(df)

    if counts.empty:
        console.print("[yellow]No tickers found[/yellow]")
        return

    # Display top tickers
    table = Table(title=f"Top {top_n} Tickers by Mentions")
    table.add_column("Ticker")
    table.add_column("Mentions", justify="right")

    for _, row in counts.head(top_n).iterrows():
        table.add_row(row["ticker"], str(row["count"]))

    console.print(table)


def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
