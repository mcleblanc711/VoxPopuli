"""Price data fetching from IBKR and yfinance."""

import logging
from datetime import date, datetime
from pathlib import Path
from typing import Literal

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class PriceFetcher:
    """Fetch and cache OHLCV price data."""

    def __init__(self, data_dir: str = "data/prices"):
        """Initialize the price fetcher.

        Args:
            data_dir: Directory to store cached price data.
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.ib = None

    def _get_cache_path(self, ticker: str) -> Path:
        """Get the cache file path for a ticker."""
        return self.data_dir / f"{ticker.upper()}_daily.parquet"

    def connect_ibkr(
        self,
        host: str = "127.0.0.1",
        port: int = 7497,
        client_id: int = 1,
    ) -> bool:
        """Connect to IBKR TWS.

        Args:
            host: TWS host
            port: TWS port (7497 for paper, 7496 for live)
            client_id: Client ID

        Returns:
            True if connection successful.
        """
        try:
            from ib_insync import IB

            self.ib = IB()
            self.ib.connect(host, port, clientId=client_id)
            logger.info(f"Connected to IBKR TWS at {host}:{port}")
            return True
        except Exception as e:
            logger.warning(f"Failed to connect to IBKR: {e}")
            self.ib = None
            return False

    def disconnect_ibkr(self) -> None:
        """Disconnect from IBKR TWS."""
        if self.ib and self.ib.isConnected():
            self.ib.disconnect()
            logger.info("Disconnected from IBKR TWS")

    def fetch_ibkr(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
        bar_size: str = "1 day",
    ) -> pd.DataFrame | None:
        """Fetch historical data from IBKR.

        Args:
            ticker: Stock ticker symbol
            start_date: Start date
            end_date: End date
            bar_size: Bar size (e.g., '1 day', '1 hour')

        Returns:
            DataFrame with OHLCV data or None if failed.
        """
        if not self.ib or not self.ib.isConnected():
            logger.warning("Not connected to IBKR")
            return None

        try:
            from ib_insync import Stock

            contract = Stock(ticker, "SMART", "USD")
            self.ib.qualifyContracts(contract)

            duration = f"{(end_date - start_date).days} D"
            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime=end_date.strftime("%Y%m%d 23:59:59"),
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow="ADJUSTED_LAST",
                useRTH=True,
            )

            if not bars:
                return None

            df = pd.DataFrame([
                {
                    "date": bar.date,
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume,
                    "adjusted_close": bar.close,
                    "source": "ibkr",
                }
                for bar in bars
            ])

            df["date"] = pd.to_datetime(df["date"]).dt.date
            return df

        except Exception as e:
            logger.error(f"IBKR fetch failed for {ticker}: {e}")
            return None

    def fetch_yfinance(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame | None:
        """Fetch historical data from yfinance.

        Args:
            ticker: Stock/crypto ticker symbol
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with OHLCV data or None if failed.
        """
        try:
            yf_ticker = yf.Ticker(ticker)
            df = yf_ticker.history(
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                auto_adjust=False,
            )

            if df.empty:
                logger.warning(f"No data returned from yfinance for {ticker}")
                return None

            df = df.reset_index()
            df = df.rename(columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
                "Adj Close": "adjusted_close",
            })

            df = df[["date", "open", "high", "low", "close", "volume", "adjusted_close"]]
            df["source"] = "yfinance"
            df["date"] = pd.to_datetime(df["date"]).dt.date

            return df

        except Exception as e:
            logger.error(f"yfinance fetch failed for {ticker}: {e}")
            return None

    def fetch(
        self,
        ticker: str,
        start_date: date | str,
        end_date: date | str,
        source: Literal["ibkr", "yfinance", "auto"] = "auto",
    ) -> pd.DataFrame | None:
        """Fetch historical data from the specified source.

        Args:
            ticker: Stock ticker symbol
            start_date: Start date (date or 'YYYY-MM-DD' string)
            end_date: End date (date or 'YYYY-MM-DD' string)
            source: Data source ('ibkr', 'yfinance', or 'auto')

        Returns:
            DataFrame with OHLCV data.
        """
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d").date()

        df = None

        if source == "ibkr" or source == "auto":
            df = self.fetch_ibkr(ticker, start_date, end_date)

        if df is None and (source == "yfinance" or source == "auto"):
            df = self.fetch_yfinance(ticker, start_date, end_date)

        return df

    def cache_prices(
        self,
        ticker: str,
        start_date: date | str,
        end_date: date | str,
        source: Literal["ibkr", "yfinance", "auto"] = "auto",
        force_refresh: bool = False,
    ) -> pd.DataFrame | None:
        """Fetch and cache price data.

        Args:
            ticker: Stock ticker symbol
            start_date: Start date
            end_date: End date
            source: Data source
            force_refresh: Force re-download even if cached

        Returns:
            DataFrame with OHLCV data.
        """
        cache_path = self._get_cache_path(ticker)

        if cache_path.exists() and not force_refresh:
            logger.info(f"Loading cached data for {ticker}")
            cached_df = pd.read_parquet(cache_path)

            # Check if we need to extend the date range
            if isinstance(start_date, str):
                start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
            if isinstance(end_date, str):
                end_date = datetime.strptime(end_date, "%Y-%m-%d").date()

            cached_start = cached_df["date"].min()
            cached_end = cached_df["date"].max()

            if cached_start <= start_date and cached_end >= end_date:
                return cached_df[
                    (cached_df["date"] >= start_date) & (cached_df["date"] <= end_date)
                ]

        # Fetch new data
        df = self.fetch(ticker, start_date, end_date, source)

        if df is not None:
            # Merge with existing cache if present
            if cache_path.exists() and not force_refresh:
                existing_df = pd.read_parquet(cache_path)
                df = pd.concat([existing_df, df]).drop_duplicates(subset=["date"])
                df = df.sort_values("date")

            df.to_parquet(cache_path, index=False)
            logger.info(f"Cached {len(df)} rows for {ticker}")

        return df

    def fetch_multiple(
        self,
        tickers: list[str],
        start_date: date | str,
        end_date: date | str,
        source: Literal["ibkr", "yfinance", "auto"] = "auto",
    ) -> dict[str, pd.DataFrame]:
        """Fetch and cache data for multiple tickers.

        Args:
            tickers: List of ticker symbols
            start_date: Start date
            end_date: End date
            source: Data source

        Returns:
            Dict mapping ticker to DataFrame.
        """
        results = {}
        for ticker in tickers:
            df = self.cache_prices(ticker, start_date, end_date, source)
            if df is not None:
                results[ticker] = df
        return results

    def load_cached(self, ticker: str) -> pd.DataFrame | None:
        """Load cached price data for a ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Cached DataFrame or None if not found.
        """
        cache_path = self._get_cache_path(ticker)
        if cache_path.exists():
            return pd.read_parquet(cache_path)
        return None

    def get_price_on_date(
        self,
        ticker: str,
        target_date: date,
        price_type: str = "adjusted_close",
    ) -> float | None:
        """Get the price for a ticker on a specific date.

        Args:
            ticker: Stock ticker symbol
            target_date: Date to get price for
            price_type: Price column to use

        Returns:
            Price or None if not available.
        """
        df = self.load_cached(ticker)
        if df is None:
            return None

        if isinstance(target_date, str):
            target_date = datetime.strptime(target_date, "%Y-%m-%d").date()

        # Find exact or closest prior date
        df = df[df["date"] <= target_date].sort_values("date", ascending=False)

        if df.empty:
            return None

        return float(df.iloc[0][price_type])
